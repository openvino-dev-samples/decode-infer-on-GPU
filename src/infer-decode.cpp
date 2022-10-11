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

#include <cldnn/cldnn_config.hpp>
#include <gpu/gpu_context_api_va.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/ocl/va.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include "util.h"

#define VERIFY(x, y)       \
    if (!(x)) {            \
        printf("%s\n", y); \
    }

#define VERIFY2(x, y)      \
    if (!(x)) {            \
        printf("%s\n", y); \
        return NULL;       \
    }

using namespace ov::preprocess;

#define BITSTREAM_BUFFER_SIZE      2000000
#define MAX_RESULTS                5
#define SYNC_TIMEOUT               60000
#define MAJOR_API_VERSION_REQUIRED 2
#define MINOR_API_VERSION_REQUIRED 2

void Usage(void) {
    printf("\n");
    printf("   Usage  :  advanced-decvpp-infer \n\n");
    printf("     -i      input file name (HEVC elementary stream)\n\n");
    printf("     -m      input model name (object detection)\n\n");
    printf("   Example:  advanced-decvpp-infer -i in.h265 -m mobilenet-ssd.xml\n");
    return;
}

mfxSession CreateVPLSession(mfxLoader *loader);
void PrintTopResults(const float *output, mfxU16 width, mfxU16 height, ov::Shape output_shape);

int main(int argc, char **argv) {
    FILE *source = NULL;

    mfxLoader loader                     = NULL;
    mfxSession session                   = NULL;
    mfxFrameSurface1 *decSurfaceOut      = NULL;
    mfxBitstream bitstream               = {};
    mfxSyncPoint syncp                   = {};
    mfxVideoParam mfxDecParams           = {};
    mfxVideoParam mfxVPPParams           = {};
    mfxFrameSurface1 *pmfxDecOutSurface  = NULL;
    mfxFrameSurface1 *pmfxVPPSurfacesOut = NULL;
    mfxU32 frameNum                      = 0;
    bool isStillGoing                    = true;
    bool isDrainingDec                   = false;
    bool isDrainingVPP                   = false;
    mfxStatus sts                        = MFX_ERR_NONE;
    Params cliParams                     = {};

    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    ov::Shape output_shape;

    VADisplay lvaDisplay;
    VASurfaceID lvaSurfaceID;
    mfxHandleType ldeviceType;
    mfxHDL lresource;
    mfxResourceType lresourceType;
    mfxStatus lsts;
    bool isNetworkLoaded = false;

    mfxU16 oriImgWidth, oriImgHeight;
    mfxU16 inputDimWidth, inputDimHeight;
    mfxU16 vppInImgWidth, vppInImgHeight;
    mfxU16 vppOutImgWidth, vppOutImgHeight;

    //-- Parse command line args to cliParams
    if (ParseArgsAndValidate(argc, argv, &cliParams, PARAMS_DECVPP) == false) {
        Usage();
        return 1; // return 1 as error code
    }

    source = fopen(cliParams.infileName, "rb");
    VERIFY(source, "Could not open input file");

    //--- Setup OpenVINO Inference Engine
    // Read network model
    model = core.read_model(cliParams.inmodelName);

    // Get the input shape
    auto input0 = model->get_parameters().at(0);
    auto shape = input0->get_shape();
    auto width = shape[3];
    auto height = shape[2];
    inputDimWidth  = (mfxU16)width;
    inputDimHeight = (mfxU16)height;

    // Get the ouput shape
    output_shape = model->output().get_shape();

    //---- Setup VPL
    // Create VPL session
    session = CreateVPLSession(&loader);
    VERIFY(session != NULL, "Not able to create VPL session");

    //-- Initialize Decode
    // Prepare input bitstream
    bitstream.MaxLength = BITSTREAM_BUFFER_SIZE;
    bitstream.Data      = (mfxU8 *)calloc(bitstream.MaxLength, sizeof(mfxU8));
    VERIFY(bitstream.Data, "Not able to allocate input buffer");
    bitstream.CodecId = MFX_CODEC_HEVC;

    sts = ReadEncodedStream(bitstream, source);
    VERIFY(MFX_ERR_NONE == sts, "Error reading bitstream");

    // Retrieve the frame information from input stream
    mfxDecParams.mfx.CodecId = MFX_CODEC_HEVC;
    mfxDecParams.IOPattern   = MFX_IOPATTERN_OUT_VIDEO_MEMORY;
    sts                      = MFXVideoDECODE_DecodeHeader(session, &bitstream, &mfxDecParams);
    VERIFY(MFX_ERR_NONE == sts, "Error decoding header");

    // Original image size
    oriImgWidth  = mfxDecParams.mfx.FrameInfo.Width;
    oriImgHeight = mfxDecParams.mfx.FrameInfo.Height;

    // Input parameters finished, now initialize decode
    sts = MFXVideoDECODE_Init(session, &mfxDecParams);
    VERIFY(MFX_ERR_NONE == sts, "Error initializing Decode");

    //-- Initialize VPP
    // Prepare vpp in/out params
    // vpp in:  decode output image size
    // vpp out: network model input size
    vppInImgWidth   = mfxDecParams.mfx.FrameInfo.Width;
    vppInImgHeight  = mfxDecParams.mfx.FrameInfo.Height;
    vppOutImgWidth  = inputDimWidth;
    vppOutImgHeight = inputDimHeight;

    mfxVPPParams.vpp.In.FourCC        = mfxDecParams.mfx.FrameInfo.FourCC;
    mfxVPPParams.vpp.In.ChromaFormat  = mfxDecParams.mfx.FrameInfo.ChromaFormat;
    mfxVPPParams.vpp.In.Width         = vppInImgWidth;
    mfxVPPParams.vpp.In.Height        = vppInImgHeight;
    mfxVPPParams.vpp.In.CropW         = vppInImgWidth;
    mfxVPPParams.vpp.In.CropH         = vppInImgHeight;
    mfxVPPParams.vpp.In.PicStruct     = MFX_PICSTRUCT_PROGRESSIVE;
    mfxVPPParams.vpp.In.FrameRateExtN = 30;
    mfxVPPParams.vpp.In.FrameRateExtD = 1;

    mfxVPPParams.vpp.Out.FourCC        = MFX_FOURCC_NV12;
    mfxVPPParams.vpp.Out.ChromaFormat  = MFX_CHROMAFORMAT_YUV420;
    mfxVPPParams.vpp.Out.Width         = ALIGN16(vppOutImgWidth);
    mfxVPPParams.vpp.Out.Height        = ALIGN16(vppOutImgHeight);
    mfxVPPParams.vpp.Out.CropW         = vppOutImgWidth;
    mfxVPPParams.vpp.Out.CropH         = vppOutImgHeight;
    mfxVPPParams.vpp.Out.PicStruct     = MFX_PICSTRUCT_PROGRESSIVE;
    mfxVPPParams.vpp.Out.FrameRateExtN = 30;
    mfxVPPParams.vpp.Out.FrameRateExtD = 1;

    mfxVPPParams.IOPattern = MFX_IOPATTERN_IN_VIDEO_MEMORY | MFX_IOPATTERN_OUT_VIDEO_MEMORY;

    sts = MFXVideoVPP_Init(session, &mfxVPPParams);
    VERIFY(MFX_ERR_NONE == sts, "Error initializing VPP");

    //-- Load network model in the shared context
    // Get the vaapi device handle
    sts = MFXVideoCORE_GetHandle(session, MFX_HANDLE_VA_DISPLAY, &lvaDisplay);
    VERIFY(MFX_ERR_NONE == sts, "MFXVideoCore_GetHandle error");


    // Integrate preprocessing steps into the execution graph with Preprocessing API
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

    // get the new inputs, one for Y and another for UV
    auto new_input0 = model->get_parameters().at(0);
    auto new_input1 = model->get_parameters().at(1);

    // Create the shared context object
    auto shared_va_context = ov::intel_gpu::ocl::VAContext(core, lvaDisplay);

    // Compile network within a shared context
    compiled_model = core.compile_model(model, shared_va_context);

    // Create infer request
    infer_request = compiled_model.create_infer_request();

    printf("Decoding VPP, and infering %s with %s\n", cliParams.infileName, cliParams.inmodelName);
    while (isStillGoing == true) {
        if (isDrainingDec == false) {
            sts = ReadEncodedStream(bitstream, source);
            if (sts != MFX_ERR_NONE)
                isDrainingDec = true;
        }

        if (!isDrainingVPP) {
            sts = MFXVideoDECODE_DecodeFrameAsync(session,
                                                  (isDrainingDec) ? NULL : &bitstream,
                                                  NULL,
                                                  &pmfxDecOutSurface,
                                                  &syncp);
        }
        else {
            sts = MFX_ERR_NONE;
        }

        switch (sts) {
            case MFX_ERR_NONE:
                sts =
                    MFXVideoVPP_ProcessFrameAsync(session, pmfxDecOutSurface, &pmfxVPPSurfacesOut);
                if (sts == MFX_ERR_NONE) {
                    sts = pmfxVPPSurfacesOut->FrameInterface->Synchronize(pmfxVPPSurfacesOut,
                                                                          SYNC_TIMEOUT);
                    VERIFY(MFX_ERR_NONE == sts, "MFXVideoCORE_SyncOperation error");

                    frameNum++;

                    sts = pmfxVPPSurfacesOut->FrameInterface->GetNativeHandle(pmfxVPPSurfacesOut,
                                                                              &lresource,
                                                                              &lresourceType);
                    VERIFY(MFX_ERR_NONE == sts, "FrameInterface->GetNativeHandle error");
                    VERIFY(MFX_RESOURCE_VA_SURFACE == lresourceType,
                           "Display device is not MFX_HANDLE_VA_DISPLAY");

                    lvaSurfaceID = *(VASurfaceID *)lresource;

                    // Wrap VPP output into remoteblobs and set it as inference input
                    auto nv12_blob = shared_va_context.create_tensor_nv12(height, width, lvaSurfaceID);
                    infer_request.set_tensor(new_input0->get_friendly_name(), nv12_blob.first);
                    infer_request.set_tensor(new_input1->get_friendly_name(), nv12_blob.second);
                    
                    // start inference on GPU
                    infer_request.start_async();
                    infer_request.wait();
                    
                    // Release surface
                    sts = pmfxVPPSurfacesOut->FrameInterface->Release(pmfxVPPSurfacesOut);
                    VERIFY(MFX_ERR_NONE == sts, "ERROR - mfxFrameSurfaceInterface->Release failed");

                    // Process output
                    auto output_tensor= infer_request.get_output_tensor(0);
                    const float *result = output_tensor.data<const float>();
                    PrintTopResults(result, oriImgWidth, oriImgHeight, output_shape);
                }
                else if (sts == MFX_ERR_MORE_DATA) {
                    if (isDrainingVPP == true)
                        isStillGoing = false;
                }
                else {
                    if (sts < 0)
                        isStillGoing = false;
                }
                break;
            case MFX_ERR_MORE_DATA:
                // The function requires more bitstream at input before decoding can proceed
                if (isDrainingDec)
                    isDrainingVPP = true;
                break;
            default:
                isStillGoing = false;
                break;
        }
    }

    printf("Decoded %d frames\n", frameNum);

    if (bitstream.Data)
        free(bitstream.Data);

    if (source)
        fclose(source);

    if (loader)
        MFXUnload(loader);

    return 0;
}

void PrintTopResults(const float *output, mfxU16 width, mfxU16 height, ov::Shape output_shape) {
    const int maxProposalCount = output_shape[2];
    const int objectSize       = output_shape[3];
    size_t batchSize           = 1;


    std::vector<std::vector<int>> boxes(batchSize);
    std::vector<std::vector<int>> classes(batchSize);

    /* Each detection has image_id that denotes processed image */
    for (int curProposal = 0; curProposal < maxProposalCount; curProposal++) {
        auto image_id = static_cast<int>(output[curProposal * objectSize + 0]);
        if (image_id < 0) {
            break;
        }

        float confidence = output[curProposal * objectSize + 2];
        auto label       = static_cast<int>(output[curProposal * objectSize + 1]);
        auto xmin        = static_cast<int>(output[curProposal * objectSize + 3] * width);
        auto ymin        = static_cast<int>(output[curProposal * objectSize + 4] * height);
        auto xmax        = static_cast<int>(output[curProposal * objectSize + 5] * width);
        auto ymax        = static_cast<int>(output[curProposal * objectSize + 6] * height);

        std::cout << "[" << curProposal << "," << label << "] element, prob = " << confidence
                  << "    (" << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")"
                  << " batch id : " << image_id;

        if (confidence > 0.5) {
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

mfxSession CreateVPLSession(mfxLoader *loader) {
    mfxStatus sts = MFX_ERR_NONE;

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
    cfgVal.Type     = MFX_VARIANT_TYPE_U32;
    cfgVal.Data.U32 = MFX_IMPL_TYPE_HARDWARE;
    sts = MFXSetConfigFilterProperty(cfg[0], (mfxU8 *)"mfxImplDescription.Impl", cfgVal);
    VERIFY2(MFX_ERR_NONE == sts, "MFXSetConfigFilterProperty failed for Impl");

    // Implementation must provide an HEVC decoder
    cfg[1] = MFXCreateConfig(*loader);
    VERIFY2(NULL != cfg[1], "MFXCreateConfig failed")
    cfgVal.Type     = MFX_VARIANT_TYPE_U32;
    cfgVal.Data.U32 = MFX_CODEC_HEVC;
    sts             = MFXSetConfigFilterProperty(
        cfg[1],
        (mfxU8 *)"mfxImplDescription.mfxDecoderDescription.decoder.CodecID",
        cfgVal);
    VERIFY2(MFX_ERR_NONE == sts, "MFXSetConfigFilterProperty failed for decoder CodecID");

    // Implementation used must have VPP scaling capability
    cfg[2] = MFXCreateConfig(*loader);
    VERIFY2(NULL != cfg[2], "MFXCreateConfig failed")
    cfgVal.Type     = MFX_VARIANT_TYPE_U32;
    cfgVal.Data.U32 = MFX_EXTBUFF_VPP_SCALING;
    sts             = MFXSetConfigFilterProperty(
        cfg[2],
        (mfxU8 *)"mfxImplDescription.mfxVPPDescription.filter.FilterFourCC",
        cfgVal);
    VERIFY2(MFX_ERR_NONE == sts, "MFXSetConfigFilterProperty failed for VPP scale");

    // Implementation used must provide API version 2.2 or newer
    cfg[3] = MFXCreateConfig(*loader);
    VERIFY2(NULL != cfg[3], "MFXCreateConfig failed")
    cfgVal.Type     = MFX_VARIANT_TYPE_U32;
    cfgVal.Data.U32 = VPLVERSION(MAJOR_API_VERSION_REQUIRED, MINOR_API_VERSION_REQUIRED);
    sts             = MFXSetConfigFilterProperty(cfg[3],
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
