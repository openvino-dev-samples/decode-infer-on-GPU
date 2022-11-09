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
#include <gpu/gpu_context_api_va.hpp>
#include <openvino/openvino.hpp>
#include "utils/functions.h"
#include "utils/util.h"
#include <gflags/gflags.h>

#define BITSTREAM_BUFFER_SIZE 2000000
#define MAX_RESULTS 5
#define SYNC_TIMEOUT 60000
#define onevpl_decode MFXVideoDECODE_DecodeFrameAsync
#define onevpl_vpp MFXVideoVPP_ProcessFrameAsync

DEFINE_string(i, "",
              "Required. Path to one input video files ");
DEFINE_string(m, "", "Required. Path to IR .xml file");

mfxSession CreateVPLSession(mfxLoader *loader);
void PrintTopResults(const float *output, mfxU16 width, mfxU16 height, ov::Shape output_shape);

int main(int argc, char **argv)
{   
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    FILE *source = NULL;
    mfxLoader loader = NULL;
    mfxSession session = NULL;
    mfxFrameSurface1 *decSurfaceOut = NULL;
    mfxBitstream bitstream = {};
    mfxSyncPoint syncp = {};
    mfxVideoParam mfxDecParams = {};
    mfxVideoParam mfxVPPParams = {};
    mfxFrameSurface1 *pmfxDecOutSurface = NULL;
    mfxFrameSurface1 *pmfxVPPSurfacesOut = NULL;
    mfxU32 frameNum = 0;
    bool isStillGoing = true;
    bool isDrainingDec = false;
    bool isDrainingVPP = false;
    mfxStatus sts = MFX_ERR_NONE;
    Params cliParams = {};

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

    source = fopen(FLAGS_i.data(), "rb");
    VERIFY(source, "Could not open input file");

    //--- Setup OpenVINO Inference Engine
    // Read network model
    model = core.read_model(FLAGS_m);

    // Get the input shape
    auto input0 = model->get_parameters().at(0);
    auto shape = input0->get_shape();
    auto width = shape[3];
    auto height = shape[2];
    inputDimWidth = (mfxU16)width;
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
    bitstream.Data = (mfxU8 *)calloc(bitstream.MaxLength, sizeof(mfxU8));
    VERIFY(bitstream.Data, "Not able to allocate input buffer");
    bitstream.CodecId = MFX_CODEC_HEVC;

    sts = ReadEncodedStream(bitstream, source);
    VERIFY(MFX_ERR_NONE == sts, "Error reading bitstream");

    // Retrieve the frame information from input stream
    mfxDecParams.mfx.CodecId = MFX_CODEC_HEVC;
    mfxDecParams.IOPattern = MFX_IOPATTERN_OUT_VIDEO_MEMORY;
    sts = MFXVideoDECODE_DecodeHeader(session, &bitstream, &mfxDecParams);
    VERIFY(MFX_ERR_NONE == sts, "Error decoding header");

    // Original image size
    oriImgWidth = mfxDecParams.mfx.FrameInfo.Width;
    oriImgHeight = mfxDecParams.mfx.FrameInfo.Height;

    // Input parameters finished, now initialize decode
    sts = MFXVideoDECODE_Init(session, &mfxDecParams);
    VERIFY(MFX_ERR_NONE == sts, "Error initializing Decode");

    //-- Initialize VPP
    // Prepare vpp in/out params
    // vpp in:  decode output image size
    // vpp out: network model input size
    vppInImgWidth = mfxDecParams.mfx.FrameInfo.Width;
    vppInImgHeight = mfxDecParams.mfx.FrameInfo.Height;
    vppOutImgWidth = inputDimWidth;
    vppOutImgHeight = inputDimHeight;

    mfxVPPParams.vpp.In.FourCC = mfxDecParams.mfx.FrameInfo.FourCC;
    mfxVPPParams.vpp.In.ChromaFormat = mfxDecParams.mfx.FrameInfo.ChromaFormat;
    mfxVPPParams.vpp.In.Width = vppInImgWidth;
    mfxVPPParams.vpp.In.Height = vppInImgHeight;
    mfxVPPParams.vpp.In.CropW = vppInImgWidth;
    mfxVPPParams.vpp.In.CropH = vppInImgHeight;
    mfxVPPParams.vpp.In.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
    mfxVPPParams.vpp.In.FrameRateExtN = 30;
    mfxVPPParams.vpp.In.FrameRateExtD = 1;

    mfxVPPParams.vpp.Out.FourCC = MFX_FOURCC_NV12;
    mfxVPPParams.vpp.Out.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
    mfxVPPParams.vpp.Out.Width = ALIGN16(vppOutImgWidth);
    mfxVPPParams.vpp.Out.Height = ALIGN16(vppOutImgHeight);
    mfxVPPParams.vpp.Out.CropW = vppOutImgWidth;
    mfxVPPParams.vpp.Out.CropH = vppOutImgHeight;
    mfxVPPParams.vpp.Out.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
    mfxVPPParams.vpp.Out.FrameRateExtN = 30;
    mfxVPPParams.vpp.Out.FrameRateExtD = 1;

    mfxVPPParams.IOPattern = MFX_IOPATTERN_IN_VIDEO_MEMORY | MFX_IOPATTERN_OUT_VIDEO_MEMORY;

    // Initialize the VPP
    sts = MFXVideoVPP_Init(session, &mfxVPPParams);
    VERIFY(MFX_ERR_NONE == sts, "Error initializing VPP");

    // Get the vaapi device handle
    sts = MFXVideoCORE_GetHandle(session, MFX_HANDLE_VA_DISPLAY, &lvaDisplay);
    VERIFY(MFX_ERR_NONE == sts, "MFXVideoCore_GetHandle error");

    // Integrate preprocessing steps into the execution graph with Preprocessing API
    openvino_preprocess(model);
    auto shared_va_context = ov::intel_gpu::ocl::VAContext(core, lvaDisplay);
    compiled_model = core.compile_model(model, shared_va_context);

    // Create infer request
    infer_request = compiled_model.create_infer_request();
    clock_t start = clock();
    printf("Decoding VPP, and infering %s with %s\n", cliParams.infileName, cliParams.inmodelName);
    while (isStillGoing == true)
    {
        if (isDrainingDec == false)
        {
            sts = ReadEncodedStream(bitstream, source);
            if (sts != MFX_ERR_NONE)
                isDrainingDec = true;
        }

        if (!isDrainingVPP)
        {   
            // Run decode with onevpl
            sts = onevpl_decode(session,
                                (isDrainingDec) ? NULL : &bitstream,
                                NULL,
                                &pmfxDecOutSurface,
                                &syncp);
        }
        else
        {
            sts = MFX_ERR_NONE;
        }

        switch (sts)
        {
        case MFX_ERR_NONE:
            // Run vpp with onevpl
            sts =
                onevpl_vpp(session, pmfxDecOutSurface, &pmfxVPPSurfacesOut);
            if (sts == MFX_ERR_NONE)
            {
                sts = pmfxVPPSurfacesOut->FrameInterface->Synchronize(pmfxVPPSurfacesOut,
                                                                      SYNC_TIMEOUT);
                VERIFY(MFX_ERR_NONE == sts, "MFXVideoCORE_SyncOperation error");

                

                sts = pmfxVPPSurfacesOut->FrameInterface->GetNativeHandle(pmfxVPPSurfacesOut,
                                                                          &lresource,
                                                                          &lresourceType);
                VERIFY(MFX_ERR_NONE == sts, "FrameInterface->GetNativeHandle error");
                VERIFY(MFX_RESOURCE_VA_SURFACE == lresourceType,
                       "Display device is not MFX_HANDLE_VA_DISPLAY");

                lvaSurfaceID = *(VASurfaceID *)lresource;

                // Wrap VPP output into remoteblobs
                auto nv12_blob = shared_va_context.create_tensor_nv12(height, width, lvaSurfaceID);

                // Run inference with openvino
                ov::Tensor result = openvino_infer(nv12_blob, model, infer_request);
                frameNum++;
                // Release surface
                sts = pmfxVPPSurfacesOut->FrameInterface->Release(pmfxVPPSurfacesOut);
                VERIFY(MFX_ERR_NONE == sts, "ERROR - mfxFrameSurfaceInterface->Release failed");

                PrintSingleResults(result, oriImgWidth, oriImgHeight);
            }
            else if (sts == MFX_ERR_MORE_DATA)
            {
                if (isDrainingVPP == true)
                    isStillGoing = false;
            }
            else
            {
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
    clock_t end = clock();
    std::cout << "Time = " << double(end - start) / CLOCKS_PER_SEC << "s" << std::endl;
    printf("Decoded %d frames\n", frameNum);

    if (bitstream.Data)
        free(bitstream.Data);

    if (source)
        fclose(source);

    if (loader)
        MFXUnload(loader);

    return 0;
}
