#include <gpu/gpu_context_api_va.hpp>
#include "blocking_queue.h"
#include <thread>
#define MAX_QUEUE_SIZE 16
#define BITSTREAM_BUFFER_SIZE 2000000
#define SYNC_TIMEOUT 600000
#define MAJOR_API_VERSION_REQUIRED 2
#define MINOR_API_VERSION_REQUIRED 2

namespace multi_source
{
    class Decode_vpp
    {
    public:
        Decode_vpp(std::vector<std::string> inputs, const ov::Shape &shape)
        {
            width = shape[3];
            height = shape[2];
            inputDimWidth = (mfxU16)width;
            inputDimHeight = (mfxU16)height;

            // Initialize the VPL session for each input source instance
            for (int i = 0; i < inputs.size(); i++)
            {
                FILE *source = NULL;
                mfxSession session = NULL;
                mfxLoader loader = NULL;
                mfxBitstream bitstream = {};

                const char *files = inputs[i].c_str();
                source = fopen(files, "rb");
                VERIFY(source, "Could not open input file");

                // Create VPL session
                session = CreateVPLSession(&loader, i);
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
                mfxU16 oriImgWidth = mfxDecParams.mfx.FrameInfo.Width;
                mfxU16 oriImgHeight = mfxDecParams.mfx.FrameInfo.Height;
                _oriImgShape.push_back(std::make_pair(oriImgHeight, oriImgWidth));

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

                _sessions.push_back(session);
                _bitstreams.push_back(bitstream);
                _sources.push_back(source);
            }
        }

        VADisplay get_context()
        {
            return lvaDisplay;
        }

        std::vector<std::pair<mfxU16, mfxU16>> get_input_shape()
        {
            return _oriImgShape;
        }

        ~Decode_vpp()
        {
            for (auto &stream : _streams)
            {
                stream.isStillGoing = false;
                stream.thread.join();
            }
        }

        void decoding(std::vector<std::string> inputs)
        {
            for (auto &input : inputs)
            {
                add_input(input);
            }
        }

        void add_input(std::string url)
        {
            // Create thread with frame reading loop
            size_t stream_id = _streams.size();
            _streams.push_back({});
            _streams.back().thread = std::thread([=]
                                                 {   
            mfxFrameSurface1 *pmfxDecOutSurface = NULL;
            mfxFrameSurface1 *pmfxVPPSurfacesOut = NULL;
            mfxSyncPoint syncp = {};

            while (_streams[stream_id].isStillGoing){
                if (_streams[stream_id].isDrainingDec == false){
                    _streams[stream_id].status = ReadEncodedStream(_bitstreams[stream_id], _sources[stream_id]);
                    VERIFY(MFX_ERR_NONE == _streams[stream_id].status, "Error reading bitstream");
                    if (_streams[stream_id].status != MFX_ERR_NONE)
                        _streams[stream_id].isDrainingDec = true;
                }

                if (!_streams[stream_id].isDrainingVPP){
                    _streams[stream_id].status = MFXVideoDECODE_DecodeFrameAsync(_sessions[stream_id],
                                                                                (_streams[stream_id].isDrainingDec) ? NULL : &_bitstreams[stream_id],
                                                                                NULL,
                                                                                &pmfxDecOutSurface,
                                                                                &syncp);
                }
                else{
                    _streams[stream_id].status = MFX_ERR_NONE;
                }

                switch (_streams[stream_id].status){
                case MFX_ERR_NONE:
                    _streams[stream_id].status =
                        MFXVideoVPP_ProcessFrameAsync(_sessions[stream_id], pmfxDecOutSurface, &pmfxVPPSurfacesOut);
                    if (_streams[stream_id].status == MFX_ERR_NONE)
                    {
                        _streams[stream_id].status = pmfxVPPSurfacesOut->FrameInterface->Synchronize(pmfxVPPSurfacesOut,
                                                                                                    SYNC_TIMEOUT);
                        VERIFY(MFX_ERR_NONE == _streams[stream_id].status, "MFXVideoCORE_SyncOperation error");

                        // wrap the VPP output and stream id into a shared surface queue
                        _queue.push(std::make_pair(pmfxVPPSurfacesOut, stream_id), MAX_QUEUE_SIZE);
                    }
                    else if (_streams[stream_id].status == MFX_ERR_MORE_DATA)
                    {
                        if (_streams[stream_id].isDrainingVPP == true)
                            _streams[stream_id].isStillGoing = false;
                    }
                    else
                    {
                        if (_streams[stream_id].status < 0)
                            _streams[stream_id].isStillGoing = false;
                    }
                    break;
                case MFX_ERR_MORE_DATA:
                    // The function requires more bitstream at input before decoding can proceed
                    if (_streams[stream_id].isDrainingDec)
                        _streams[stream_id].isDrainingVPP = true;
                    break;
                }
            }
            _streams[stream_id].isStillGoing = false; });
        }

        std::pair<mfxFrameSurface1 *, size_t> read()
        {
            return _queue.pop();
        }

        mfxSession CreateVPLSession(mfxLoader *loader, int counter)
        {
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

            // share one vaapi device handle amonge different input source
            if (counter == 0)
            {
                VADisplay va_dpy = NULL;
                int fd;
                // initialize VAAPI context and set session handle (req in Linux)
                fd = open("/dev/dri/renderD128", O_RDWR);
                if (fd >= 0)
                {
                    va_dpy = vaGetDisplayDRM(fd);
                    if (va_dpy)
                    {
                        int major_version = 0, minor_version = 0;
                        if (VA_STATUS_SUCCESS == vaInitialize(va_dpy, &major_version, &minor_version))
                        {
                            sts = MFXVideoCORE_SetHandle(session,
                                                         static_cast<mfxHandleType>(MFX_HANDLE_VA_DISPLAY),
                                                         va_dpy);
                            VERIFY(MFX_ERR_NONE == sts, "SetHandle error");
                        }
                    }
                }
            }
            else
            {
                sts = MFXVideoCORE_SetHandle(session,
                                             static_cast<mfxHandleType>(MFX_HANDLE_VA_DISPLAY),
                                             lvaDisplay);
                VERIFY(MFX_ERR_NONE == sts, "SetHandle error");
            }

            // Print info about implementation loaded
            ShowImplementationInfo(*loader, 0);
            return session;
        }

    private:
        BlockingQueue<std::pair<mfxFrameSurface1 *, size_t>> _queue;
        struct StreamState
        {
            bool isStillGoing = true;
            bool isDrainingDec = false;
            bool isDrainingVPP = false;
            mfxStatus status = MFX_ERR_NONE;
            std::thread thread;
        };
        std::vector<StreamState> _streams;
        std::vector<mfxSession> _sessions;
        std::vector<mfxBitstream> _bitstreams;
        std::vector<FILE *> _sources;
        std::vector<std::pair<mfxU16, mfxU16>> _oriImgShape;
        size_t width;
        size_t height;

        mfxStatus sts = MFX_ERR_NONE;
        bool isNetworkLoaded = false;
        mfxU16 inputDimWidth, inputDimHeight;
        mfxU16 vppInImgWidth, vppInImgHeight;
        mfxU16 vppOutImgWidth, vppOutImgHeight;
        mfxVideoParam mfxDecParams = {};
        mfxVideoParam mfxVPPParams = {};
        VADisplay lvaDisplay;
    };
}