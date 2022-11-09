#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/ocl/va.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include "utils/util.h"

using namespace ov::preprocess;

mfxStatus sts = MFX_ERR_NONE;
#define MAJOR_API_VERSION_REQUIRED 2
#define MINOR_API_VERSION_REQUIRED 2

bool openvino_preprocess(std::shared_ptr<ov::Model> model)
{
    auto p = PrePostProcessor(model);
    p.input().tensor().set_element_type(ov::element::u8)
        // YUV images can be split into separate planes
        .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
        .set_memory_type(ov::intel_gpu::memory_type::surface);
    // Change color format
    p.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
    // Change layout
    p.input().model().set_layout("NCHW");
    model = p.build();
    return true;
}

ov::Tensor openvino_infer(std::pair<ov::intel_gpu::ocl::VASurfaceTensor, ov::intel_gpu::ocl::VASurfaceTensor> nv12_blob, std::shared_ptr<ov::Model> &model, ov::InferRequest infer_request)
{   
    // get the new inputs, one for Y and another for UV
    auto new_input0 = model->get_parameters().at(0);
    auto new_input1 = model->get_parameters().at(1);

    // set remoteblobs as inference input
    infer_request.set_tensor(new_input0->get_friendly_name(), nv12_blob.first);
    infer_request.set_tensor(new_input1->get_friendly_name(), nv12_blob.second);

    // start inference on GPU
    infer_request.start_async();
    infer_request.wait();
    // Process output
    auto output_tensor = infer_request.get_output_tensor(0);
    return output_tensor;
}

void PrintSingleResults(ov::Tensor output_tensor, mfxU16 width, mfxU16 height)
{
    float *output = (float *)output_tensor.data();
    /* Each detection has image_id that denotes processed image */
    size_t last_dim = output_tensor.get_shape().back();
    if (last_dim == 7)
    {
        // suppose object detection model with output [image_id, label_id, confidence, bbox coordinates]
        float *output = (float *)output_tensor.data();
        for (size_t i = 0; i < output_tensor.get_size() / last_dim; i++)
        {
            int image_id = static_cast<int>(output[i * last_dim + 0]);
            if (image_id < 0)
                break;

            float confidence = output[i * last_dim + 2];

            if (confidence < 0.5)
            {
                continue;
            }

            float x_min = output[i * last_dim + 3] * width;
            float y_min = output[i * last_dim + 4] * height;
            float x_max = output[i * last_dim + 5] * width;
            float y_max = output[i * last_dim + 6] * height;
            printf("  image%d: bbox %.2f, %.2f, %.2f, %.2f, confidence = %.5f\n", image_id, x_min, y_min, x_max, y_max,
                   confidence);
        }
    }
    else
    {
        std::cout << "  output shape=" << output_tensor.get_shape() << std::endl;
    }
}

void PrintMultiResults(ov::Tensor output_tensor, std::vector<std::pair<mfxFrameSurface1*, size_t>> batched_frames, std::vector<std::pair<mfxU16, mfxU16>> shape)
{
    printf("Frames");
    for (auto frame : batched_frames)
    {
        printf(" [stream_id=%ld]", frame.second);
    }
    printf("\n");
    // If object detection model, print bounding box coordinates and confidence. Otherwise, print output shape
    size_t last_dim = output_tensor.get_shape().back();
    if (last_dim == 7)
    {
        // suppose object detection model with output [image_id, label_id, confidence, bbox coordinates]
        float *output = (float *)output_tensor.data();
        for (size_t i = 0; i < output_tensor.get_size() / last_dim; i++)
        {
            int image_id = static_cast<int>(output[i * last_dim + 0]);
            int batch_id = batched_frames[image_id].second;
            if (image_id < 0)
                break;

            float confidence = output[i * last_dim + 2];

            if (confidence < 0.5)
            {
                continue;
            }

            float x_min = output[i * last_dim + 3] * shape[batch_id].second;
            float y_min = output[i * last_dim + 4] * shape[batch_id].first;
            float x_max = output[i * last_dim + 5] * shape[batch_id].second;
            float y_max = output[i * last_dim + 6] * shape[batch_id].first;
            printf("image%d: bbox %.2f, %.2f, %.2f, %.2f, confidence = %.5f\n", image_id, x_min, y_min, x_max, y_max,
                   confidence);
        }
    }
    else
    {
        std::cout << "  output shape=" << output_tensor.get_shape() << std::endl;
    }
}


mfxSession CreateVPLSession(mfxLoader *loader)
{

    // variables used only in 2.x version
    mfxConfig cfg[4];
    mfxVariant cfgVal;
    mfxSession session = NULL;

    //-- Create session
    *loader = MFXLoad();
    VERIFY2(NULL != *loader, "MFXLoad failed -- is implementation in path?\n");

    // Implementation used must be the hardware implementation
    cfg[0] = MFXCreateConfig(*loader);
    VERIFY2(NULL != cfg[0], "MFXCreateConfig failed")
    cfgVal.Type = MFX_VARIANT_TYPE_U32;
    cfgVal.Data.U32 = MFX_IMPL_TYPE_HARDWARE;
    sts = MFXSetConfigFilterProperty(cfg[0], (mfxU8 *)"mfxImplDescription.Impl", cfgVal);
    VERIFY2(MFX_ERR_NONE == sts, "MFXSetConfigFilterProperty failed for Impl");

    // Implementation must provide an HEVC decoder
    cfg[1] = MFXCreateConfig(*loader);
    VERIFY2(NULL != cfg[1], "MFXCreateConfig failed")
    cfgVal.Type = MFX_VARIANT_TYPE_U32;
    cfgVal.Data.U32 = MFX_CODEC_HEVC;
    sts = MFXSetConfigFilterProperty(
        cfg[1],
        (mfxU8 *)"mfxImplDescription.mfxDecoderDescription.decoder.CodecID",
        cfgVal);
    VERIFY2(MFX_ERR_NONE == sts, "MFXSetConfigFilterProperty failed for decoder CodecID");

    // Implementation used must have VPP scaling capability
    cfg[2] = MFXCreateConfig(*loader);
    VERIFY2(NULL != cfg[2], "MFXCreateConfig failed")
    cfgVal.Type = MFX_VARIANT_TYPE_U32;
    cfgVal.Data.U32 = MFX_EXTBUFF_VPP_SCALING;
    sts = MFXSetConfigFilterProperty(
        cfg[2],
        (mfxU8 *)"mfxImplDescription.mfxVPPDescription.filter.FilterFourCC",
        cfgVal);
    VERIFY2(MFX_ERR_NONE == sts, "MFXSetConfigFilterProperty failed for VPP scale");

    // Implementation used must provide API version 2.2 or newer
    cfg[3] = MFXCreateConfig(*loader);
    VERIFY2(NULL != cfg[3], "MFXCreateConfig failed")
    cfgVal.Type = MFX_VARIANT_TYPE_U32;
    cfgVal.Data.U32 = VPLVERSION(MAJOR_API_VERSION_REQUIRED, MINOR_API_VERSION_REQUIRED);
    sts = MFXSetConfigFilterProperty(cfg[3],
                                     (mfxU8 *)"mfxImplDescription.ApiVersion.Version",
                                     cfgVal);
    VERIFY2(MFX_ERR_NONE == sts, "MFXSetConfigFilterProperty failed for API version");

    sts = MFXCreateSession(*loader, 0, &session);
    VERIFY2(MFX_ERR_NONE == sts,
            "Cannot create session -- no implementations meet selection criteria");

    // Print info about implementation loaded
    ShowImplementationInfo(*loader, 0);
    return session;
}
