#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/ocl/va.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include "util.h"

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

float* openvino_infer(std::pair<ov::intel_gpu::ocl::ClImage2DTensor, ov::intel_gpu::ocl::ClImage2DTensor> nv12_blob, std::shared_ptr<ov::Model> &model, ov::InferRequest infer_request)
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
    float *result = output_tensor.data<float>();
    return result;
}

void PrintTopResults(const float *output, mfxU16 width, mfxU16 height, ov::Shape output_shape)
{
    const int maxProposalCount = output_shape[2];
    const int objectSize = output_shape[3];
    size_t batchSize = 1;

    std::vector<std::vector<int>> boxes(batchSize);
    std::vector<std::vector<int>> classes(batchSize);

    /* Each detection has image_id that denotes processed image */
    for (int curProposal = 0; curProposal < maxProposalCount; curProposal++)
    {
        auto image_id = static_cast<int>(output[curProposal * objectSize + 0]);
        if (image_id < 0)
        {
            break;
        }

        float confidence = output[curProposal * objectSize + 2];
        auto label = static_cast<int>(output[curProposal * objectSize + 1]);
        auto xmin = static_cast<int>(output[curProposal * objectSize + 3] * width);
        auto ymin = static_cast<int>(output[curProposal * objectSize + 4] * height);
        auto xmax = static_cast<int>(output[curProposal * objectSize + 5] * width);
        auto ymax = static_cast<int>(output[curProposal * objectSize + 6] * height);

        std::cout << "[" << curProposal << "," << label << "] element, prob = " << confidence
                  << "    (" << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")"
                  << " batch id : " << image_id;

        if (confidence > 0.5)
        {
            /** Drawing only objects with >50% probability **/
            classes[image_id].push_back(label);
            boxes[image_id].push_back(xmin);
            boxes[image_id].push_back(ymin);
            boxes[image_id].push_back(xmax - xmin);
            boxes[image_id].push_back(ymax - ymin);
            std::cout << " WILL BE PRINTED!";
        }
        std::cout << std::endl;
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
