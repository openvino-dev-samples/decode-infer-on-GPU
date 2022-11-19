//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

///
/// A minimal oneAPI Video Processing Library (oneVPL) decode, vpp and infer application,
/// using 2.x API with internal memory management,
/// showing zerocopy with remoteblob
///
/// @file

#include "time.h"
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <memory>
#include <gflags/gflags.h>
#include <thread>
#include "utils/functions.h"
#include "utils/util.h"
#include "decode_vpp.h"
#include "blocking_queue.h"

using namespace multi_source;
DEFINE_string(i, "",
              "Required. Path to one or multiple input video files "
              "(separated by comma or delimeter specified in -delimeter option");
DEFINE_string(m, "", "Required. Path to IR .xml file");
// DEFINE_string(device, "GPU", "Device for decode and inference, 'CPU' or 'GPU'");
DEFINE_int32(bs, 2, "Batch size");
DEFINE_int32(nr, 4, "Number of inference requests");
DEFINE_int32(ns, 1, "Number of GPU streams");
DEFINE_int32(fr, 30, "Number of frame to be decoded for each input source");

int main(int argc, char *argv[])
{

    gflags::ParseCommandLineFlags(&argc, &argv, true);
    int frameNum = 0;

    // setup OpenVINO Inference Engine
    ov::Core core;
    
    // configuration for Multiple streams on GPU
    std::string key = "GPU_THROUGHPUT_STREAMS";
    ov::AnyMap config;
    config[key] = FLAGS_ns;
    core.set_property("GPU", config);

    // read network model
    std::shared_ptr<ov::Model> model = core.read_model(FLAGS_m);

    // get the input shape
    auto input0 = model->get_parameters().at(0);
    auto shape = input0->get_shape();
    auto inputs = split_string(FLAGS_i);
    int num_source = inputs.size();

    // setup VPL
    Decode_vpp decode_vpp(inputs, shape);
    auto lvaDisplay = decode_vpp.get_context();
    auto input_shape = decode_vpp.get_input_shape();

    // integrate preprocessing steps into the execution graph with Preprocessing API
    openvino_preprocess(model);

    if (FLAGS_bs > 1)
        ov::set_batch(model, FLAGS_bs);
    // zero-copy conversion from VAAPI surface to OpenVINO toolkit tensors (one for Y plane, another for UV)
    auto shared_va_context = ov::intel_gpu::ocl::VAContext(core, lvaDisplay);
    ov::CompiledModel compiled_model = core.compile_model(model, shared_va_context);

    // create the queue for free infer request
    BlockingQueue<ov::InferRequest> free_requests;
    for (int i = 0; i < FLAGS_nr; i++)
        free_requests.push(compiled_model.create_infer_request());

    clock_t start = clock();

    // reading the input data and start decoding
    decode_vpp.decoding(inputs);
    BlockingQueue<std::pair<std::vector<std::pair<mfxFrameSurface1 *, size_t>>, ov::InferRequest>> busy_requests;

    // async thread waiting for inference completion and printing inference results
    std::thread thread([&]
                       {
        for (;;) {
            auto res = busy_requests.pop();
            auto batched_frames = res.first;
            auto infer_request = res.second;
            if (!infer_request)
                break;
            infer_request.wait();
            ov::Tensor output_tensor = infer_request.get_output_tensor(0);
            PrintMultiResults(output_tensor, batched_frames, input_shape);
            // When application completes the work with frame surface, it must call release to avoid memory leaks
            for (auto frame : batched_frames)
            {
                frame.first->FrameInterface->Release(frame.first);
            }
            free_requests.push(infer_request);
        }
        printf("print_tensor() thread completed\n"); });

    int inferedNum = 0;
    // frame loop
    std::vector<std::pair<mfxFrameSurface1 *, size_t>> batched_frames;
    
    for (;;)
    {
        inferedNum++;
        if (inferedNum > FLAGS_fr * num_source) // End-Of-Stream or error
            break;
        // video input, decode, resize
        auto frame = decode_vpp.read();

        // fill full batch
        batched_frames.push_back(frame);
        if (batched_frames.size() < FLAGS_bs)
            continue;

        // zero-copy conversion from VASurfaceID to OpenVINO VASurfaceTensor (one tensor for Y plane, another for UV)
        std::vector<ov::Tensor> y_tensors;
        std::vector<ov::Tensor> uv_tensors;
        for (auto va_surface : batched_frames)
        {
            mfxResourceType lresourceType;
            mfxHDL lresource;
            va_surface.first->FrameInterface->GetNativeHandle(va_surface.first,
                                                              &lresource,
                                                              &lresourceType);
            VASurfaceID lvaSurfaceID = *(VASurfaceID *)(lresource);
            auto nv12_tensor = shared_va_context.create_tensor_nv12(shape[2], shape[3], lvaSurfaceID);
            y_tensors.push_back(nv12_tensor.first);
            uv_tensors.push_back(nv12_tensor.second);
        }

        // get inference request and start asynchronously
        ov::InferRequest infer_request = free_requests.pop();
        infer_request.set_input_tensors(0, y_tensors);  // first input is batch of Y planes
        infer_request.set_input_tensors(1, uv_tensors); // second input is batch of UV planes
        infer_request.start_async();
        busy_requests.push({batched_frames, infer_request});

        batched_frames.clear();
    }

    // wait for all inference requests in queue
    busy_requests.push({});
    thread.join();
    printf("decoded and infered %d frames\n", inferedNum - 1);
    clock_t end = clock();
    std::cout << "Time = " << double(end - start) / CLOCKS_PER_SEC << "s" << std::endl;
    return 0;
}