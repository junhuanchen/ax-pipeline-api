
#include <pybind11/pybind11.h>

#include <opencv2/opencv.hpp>

#include "axdl/include/c_api.h"

#include "common/common_func.h"
#include "common/common_pipeline.h"

#include "utilities/sample_log.h"

#include "ax_ivps_api.h"
#include "npu_cv_kit/ax_npu_imgproc.h"

#include "signal.h"

#include <queue>

#include <fstream>


namespace py = pybind11;

#define CALC_FPS(tips)                                                                                     \
  {                                                                                                        \
    static int fcnt = 0;                                                                                   \
    fcnt++;                                                                                                \
    static struct timespec ts1, ts2;                                                                       \
    clock_gettime(CLOCK_MONOTONIC, &ts2);                                                                  \
    if ((ts2.tv_sec * 1000 + ts2.tv_nsec / 1000000) - (ts1.tv_sec * 1000 + ts1.tv_nsec / 1000000) >= 1000) \
    {                                                                                                      \
      printf("%s => H26X FPS:%d\n", tips, fcnt);                                                  \
      ts1 = ts2;                                                                                           \
      fcnt = 0;                                                                                            \
    }                                                                                                      \
  }

struct _g_m3axpi_
{
    static AX_S32 g_isp_force_loop_exit;
    static CAMERA_T gCams[MAX_CAMERAS];
    static pthread_mutex_t g_forward_mutex;
    static pthread_mutex_t g_capture_mutex;
    static cv::Mat g_capture, g_display;
    static std::queue<axdl_results_t> g_result_forward;
    static pipeline_t pipelines[4];
    static int bRunJoint, bRunState;
    static void *gNpuModels;
    static int sUserWidth, sUserHeight;

    int SAMPLE_MAJOR_STREAM_WIDTH, SAMPLE_MAJOR_STREAM_HEIGHT;
    int SAMPLE_IVPS_ALGO_WIDTH, SAMPLE_IVPS_ALGO_HEIGHT;

    AX_S32 sFramerate = 30;
    COMMON_SYS_CASE_E eSysCase = SYS_CASE_SINGLE_GC4653;
    AX_SNS_HDR_MODE_E eHdrMode = AX_SNS_LINEAR_MODE;
    COMMON_SYS_ARGS_T tCommonArgs = {0};

    _g_m3axpi_() {
        init();
        load();
    }

    ~_g_m3axpi_() {
        exit();
    }

    void init()
    {
        exit();

        if (bRunState == 0) {

            memset(gCams, 0, sizeof(gCams));
            memset(&tCommonArgs, 0, sizeof(tCommonArgs));
            g_isp_force_loop_exit = 0;
            bRunJoint = 0;
            bRunState = 0;
            SAMPLE_IVPS_ALGO_WIDTH = 960;
            SAMPLE_IVPS_ALGO_HEIGHT = 540;

            AX_S32 s32Ret = 0;
            SAMPLE_SNS_TYPE_E eSnsType;
            signal(SIGPIPE, SIG_IGN);

            ALOGN("eSysCase=%d,eHdrMode=%d\n", eSysCase, eHdrMode);

            s32Ret = COMMON_SET_CAM(gCams, eSysCase, eHdrMode, &eSnsType, &tCommonArgs, sFramerate);
            if (s32Ret)
            {
                bRunState = 1;
                ALOGE("COMMON_SET_CAM failed,s32Ret:0x%x\n", s32Ret);
                exit();
                return;
            }

            SAMPLE_MAJOR_STREAM_WIDTH = gCams[0].stChnAttr.tChnAttr[AX_YUV_SOURCE_ID_MAIN].nWidth;
            SAMPLE_MAJOR_STREAM_HEIGHT = gCams[0].stChnAttr.tChnAttr[AX_YUV_SOURCE_ID_MAIN].nHeight;

            /*step 1:sys init*/
            s32Ret = COMMON_SYS_Init(&tCommonArgs);
            if (s32Ret)
            {
                bRunState = 1;
                ALOGE("COMMON_SYS_Init failed,s32Ret:0x%x\n", s32Ret);
                exit();
                return;
            }

            /*step 2:npu init*/
            AX_NPU_SDK_EX_ATTR_T sNpuAttr;
            sNpuAttr.eHardMode = AX_NPU_VIRTUAL_1_1;
            s32Ret = AX_NPU_SDK_EX_Init_with_attr(&sNpuAttr);
            if (s32Ret)
            {
                bRunState = 1;
                ALOGE("AX_NPU_SDK_EX_Init_with_attr failed,s32Ret:0x%x\n", s32Ret);
                exit();
                return;
            }

            /*step 3:camera init*/
            s32Ret = COMMON_CAM_Init();
            if (0 != s32Ret)
            {
                bRunState = 2;
                ALOGE("COMMON_CAM_Init failed,s32Ret:0x%x\n", s32Ret);
                exit();
                return;
            }

            for (int i = 0; i < tCommonArgs.nCamCnt; i++)
            {
                s32Ret = COMMON_CAM_Open(&gCams[i]);
                if (s32Ret)
                {
                    bRunState = 3;
                    ALOGE("COMMON_CAM_Open failed,s32Ret:0x%x\n", s32Ret);
                }
                gCams[i].bOpen = AX_TRUE;
                ALOGN("camera %d is open\n", i);
            }

            for (AX_S32 i = 0; i < MAX_CAMERAS; i++)
            {
                if (gCams[i].bOpen) {
                    pthread_create(&gCams[i].tIspProcThread, NULL, IspRun, (AX_VOID *)i);
                }
            }

            memset(&pipelines[0], 0, sizeof(pipelines));

            pipeline_t &pipe0 = pipelines[0];
            {
                pipeline_ivps_config_t &config0 = pipe0.m_ivps_attr;
                config0.n_ivps_grp = 0;    // 重复的会创建失败
                config0.n_ivps_fps = 60;   // 屏幕只能是60gps
                config0.n_ivps_rotate = 1; // 旋转
                config0.n_ivps_width = 854;
                config0.n_ivps_height = 480;
                config0.n_osd_rgn = 1; // osd rgn 的个数，一个rgn可以osd 32个目标
            }
            pipe0.enable = 1;
            pipe0.pipeid = 0x90015;
            pipe0.m_input_type = pi_vin;
            pipe0.m_output_type = po_vo_sipeed_maix3_screen;
            pipe0.n_loog_exit = 0; // 可以用来控制线程退出（如果有的话）
            pipe0.n_vin_pipe = 0;
            pipe0.n_vin_chn = 0;
            create_pipeline(&pipelines[0]);

            pipeline_t &pipe1 = pipelines[1];
            {
                pipeline_ivps_config_t &config1 = pipe1.m_ivps_attr;
                config1.n_ivps_grp = 1; // 重复的会创建失败
                config1.n_ivps_fps = 60;
                config1.n_ivps_width = sUserWidth;
                config1.n_ivps_height = sUserHeight;
                config1.n_fifo_count = 1; // 如果想要拿到数据并输出到回调 就设为1~4
            }
            pipe1.enable = 1;
            pipe1.pipeid = 0x90016;
            pipe1.m_input_type = pi_vin;
            pipe1.m_output_type = po_buff_bgr;
            pipe1.n_loog_exit = 0;
            pipe1.n_vin_pipe = 0;
            pipe1.n_vin_chn = 0;
            pipe1.output_func = vi_inference_func; // 图像输出的回调函数
            create_pipeline(&pipelines[1]);

            bRunState = 4;
        }

    }

    void exit()
    {
        if (bRunState) {

            // 销毁pipeline
            {
                drop();

                pipeline_t &pipe0 = pipelines[0];
                destory_pipeline(&pipe0);

                pipeline_t &pipe1 = pipelines[1];
                pipe1.output_func = NULL;
                destory_pipeline(&pipe1);

            }

            if (bRunState > 3) {

                g_isp_force_loop_exit = 1;

                for (AX_S32 i = 0; i < MAX_CAMERAS; i++)
                {
                    if (gCams[i].bOpen)
                    {
                        pthread_cancel(gCams[i].tIspProcThread);
                        AX_S32 s32Ret = pthread_join(gCams[i].tIspProcThread, NULL);
                        if (s32Ret < 0)
                        {
                            ALOGE(" isp run thread exit failed,s32Ret:0x%x\n", s32Ret);
                        }
                    }
                }
            }

            if (bRunState > 2) {
                for (AX_S32 i = 0; i < tCommonArgs.nCamCnt; i++)
                {
                    if (!gCams[i].bOpen)
                        continue;
                    COMMON_CAM_Close(&gCams[i]);
                }
            }

            if (bRunState > 1) {
                COMMON_CAM_Deinit();
            }

            if (bRunState > 0) {
                COMMON_SYS_DeInit();
            }

            bRunState = 0;
        }

    }

    int load(std::string config_file = "/home/config/yolov5s.json")
    {
        drop();
        if (bRunJoint == 0 && std::ifstream(config_file.c_str()).good()) {

            gNpuModels = NULL;

            AX_S32 s32Ret = 0;

            s32Ret = axdl_parse_param_init((char *)config_file.c_str(), &gNpuModels);

            if (s32Ret != 0)
            {
                ALOGE("sample_parse_param_det failed");
                bRunJoint = 0;
                return bRunJoint;
            }
            else
            {
                s32Ret = axdl_get_ivps_width_height(gNpuModels, (char *)config_file.c_str(), &SAMPLE_IVPS_ALGO_WIDTH, &SAMPLE_IVPS_ALGO_HEIGHT);
                ALOGI("IVPS AI channel width=%d heighr=%d", SAMPLE_IVPS_ALGO_WIDTH, SAMPLE_IVPS_ALGO_HEIGHT);
                bRunJoint = 1;
            }

            pipeline_t &pipe2 = pipelines[2];
            {
                pipeline_ivps_config_t &config2 = pipe2.m_ivps_attr;
                config2.n_ivps_grp = 2; // 重复的会创建失败
                config2.n_ivps_fps = 60;
                config2.n_ivps_width = SAMPLE_IVPS_ALGO_WIDTH;
                config2.n_ivps_height = SAMPLE_IVPS_ALGO_HEIGHT;
                if (axdl_get_model_type(gNpuModels) != MT_SEG_PPHUMSEG)
                {
                    config2.b_letterbox = 1;
                }
                config2.n_fifo_count = 1; // 如果想要拿到数据并输出到回调 就设为1~4
            }
            pipe2.enable = 1;
            pipe2.pipeid = 0x90017;
            pipe2.m_input_type = pi_vin;

            if (gNpuModels && bRunJoint)
            {
                switch (axdl_get_color_space(gNpuModels))
                {
                case axdl_color_space_rgb:
                    pipe2.m_output_type = po_buff_rgb;
                    break;
                case axdl_color_space_bgr:
                    pipe2.m_output_type = po_buff_bgr;
                    break;
                case axdl_color_space_nv12:
                default:
                    pipe2.m_output_type = po_buff_nv12;
                    break;
                }
            }
            else
            {
                pipe2.enable = 0;
                bRunJoint = 1;
                drop();
                return bRunJoint;
            }

            pipe2.n_loog_exit = 0;
            pipe2.n_vin_pipe = 0;
            pipe2.n_vin_chn = 0;
            pipe2.output_func = ai_inference_func; // 图像输出的回调函数

            create_pipeline(&pipelines[2]);

            bRunJoint = 2;
        } else {
            printf("\r\n[m3axpi.load]( check json file path : %s )\r\n\r\n", config_file.c_str());
        }

        return bRunJoint;
    }

    void drop()
    {
        if (bRunJoint) {

            if (bRunJoint > 1) {

                pipelines[2].output_func = NULL;
                destory_pipeline(&pipelines[2]);

                if (gNpuModels) {
                    axdl_deinit(&gNpuModels);
                    gNpuModels = NULL;
                }
            }

            bRunJoint = 0;

        }
    }

    static void *IspRun(void *args)
    {
        AX_U32 i = (AX_U32)args;

        ALOGN("cam %d is running...\n", i);

        while (!g_isp_force_loop_exit)
        {
            if (!gCams[i].bOpen)
            {
                usleep(40 * 1000);
                continue;
            }

            AX_ISP_Run(gCams[i].nPipeId);
        }
        return NULL;
    }

    static void vi_inference_func(pipeline_buffer_t *buff)
    {
        if (bRunState)
        {
            cv::Mat img(buff->n_height, buff->n_width, CV_8UC3, buff->p_vir);
            pthread_mutex_lock(&g_capture_mutex);
            g_capture = img.clone();
            pthread_mutex_unlock(&g_capture_mutex);
            // CALC_FPS("vi_inference_func");
        }
    }

    static void ai_inference_func(pipeline_buffer_t *buff)
    {
        if (bRunJoint)
        {
            static axdl_results_t mResults;
            axdl_image_t tSrcFrame = {0};
            switch (buff->d_type)
            {
                case po_buff_nv12:
                    tSrcFrame.eDtype = axdl_color_space_nv12;
                    break;
                case po_buff_bgr:
                    tSrcFrame.eDtype = axdl_color_space_bgr;
                    break;
                case po_buff_rgb:
                    tSrcFrame.eDtype = axdl_color_space_rgb;
                    break;
                default:
                    break;
            }
            tSrcFrame.nWidth = buff->n_width;
            tSrcFrame.nHeight = buff->n_height;
            tSrcFrame.pVir = (unsigned char *)buff->p_vir;
            tSrcFrame.pPhy = buff->p_phy;
            tSrcFrame.tStride_W = buff->n_stride;
            tSrcFrame.nSize = buff->n_size;

            axdl_inference(gNpuModels, &tSrcFrame, &mResults);
            pthread_mutex_lock(&g_forward_mutex);
            if (g_result_forward.size()) g_result_forward.pop();
            g_result_forward.push(mResults);
            pthread_mutex_unlock(&g_forward_mutex);
            // CALC_FPS("ai_inference_func");
        }
    }

} g_m3axpi;

AX_S32 _g_m3axpi_::g_isp_force_loop_exit = 0;
CAMERA_T _g_m3axpi_::gCams[MAX_CAMERAS];
pthread_mutex_t _g_m3axpi_::g_forward_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t _g_m3axpi_::g_capture_mutex = PTHREAD_MUTEX_INITIALIZER;
cv::Mat _g_m3axpi_::g_capture, _g_m3axpi_::g_display;
std::queue<axdl_results_t> _g_m3axpi_::g_result_forward;
pipeline_t _g_m3axpi_::pipelines[4];
int _g_m3axpi_::bRunJoint = 0, _g_m3axpi_::bRunState = 0;
void *_g_m3axpi_::gNpuModels = NULL;
int _g_m3axpi_::sUserWidth = 640;
int _g_m3axpi_::sUserHeight = 360;

static void g_m3axpi_camera(int CameraWidth, int CameraHeight, int SysCase, int HdrMode, int FrameRate)
{
    g_m3axpi.sUserWidth = CameraWidth;
    g_m3axpi.sUserHeight = CameraHeight;
    g_m3axpi.eSysCase = (COMMON_SYS_CASE_E)SysCase;
    g_m3axpi.eHdrMode = (AX_SNS_HDR_MODE_E)HdrMode;
    g_m3axpi.sFramerate = FrameRate;
    g_m3axpi.init();
}

static void g_m3axpi_load(py::str CfgPath)
{
    g_m3axpi.load(CfgPath.cast<std::string>());
}

static void g_m3axpi_display(py::list img)
{
    if (img.size() != 4) return;
    int rows = img[0].cast<int>();
    int cols = img[1].cast<int>();
    int channel = img[2].cast<int>();
    if (!g_m3axpi.bRunState || rows == 0 || cols == 0) return;

    AX_IMG_FORMAT_E format;
    int type;
    if (channel == 3) {
        format = AX_FORMAT_RGB888;
        type = CV_8UC3;
    }
    else if (channel == 4)
    {
        format = AX_FORMAT_RGBA8888;
        type = CV_8UC4;
    }
    else
    {
        return; // puts("(3, 4), (CV_8UC3|CV_8UC4), (AX_FORMAT_RGB888|AX_FORMAT_RGB888)")
    }

    std::string tmp = img[3].cast<std::string>();
    cv::Mat src(rows, cols, type, (void *)tmp.c_str());

    auto osd_pipe = &g_m3axpi.pipelines[0];
    axdl_canvas_t img_overlay = { src.data, src.cols, src.rows, channel };
    AX_IVPS_RGN_DISP_GROUP_S tDisp;

    if (channel == 4)
    {
        for (uint32_t *rgba2abgr = (uint32_t *)src.data, i = 0, s = src.cols*src.rows; i != s; i++)
        {
            rgba2abgr[i] = __builtin_bswap32(rgba2abgr[i]);
        }
    }

    tDisp.nNum = 1;
    tDisp.tChnAttr.nAlpha = 1024;
    tDisp.tChnAttr.eFormat = format;
    tDisp.tChnAttr.nZindex = 1;
    tDisp.tChnAttr.nBitColor.nColor = 0xFF0000;
    tDisp.tChnAttr.nBitColor.bEnable = AX_FALSE;
    tDisp.tChnAttr.nBitColor.nColorInv = 0xFF;
    tDisp.tChnAttr.nBitColor.nColorInvThr = 0xA0A0A0;

    tDisp.arrDisp[0].bShow = AX_TRUE;
    tDisp.arrDisp[0].eType = AX_IVPS_RGN_TYPE_OSD;

    tDisp.arrDisp[0].uDisp.tOSD.bEnable = AX_TRUE;
    tDisp.arrDisp[0].uDisp.tOSD.enRgbFormat = format;
    tDisp.arrDisp[0].uDisp.tOSD.u32Zindex = 1;
    tDisp.arrDisp[0].uDisp.tOSD.u32ColorKey = 0x0;
    tDisp.arrDisp[0].uDisp.tOSD.u32BgColorLo = 0xFFFFFFFF;
    tDisp.arrDisp[0].uDisp.tOSD.u32BgColorHi = 0xFFFFFFFF;
    tDisp.arrDisp[0].uDisp.tOSD.u32BmpWidth = img_overlay.width;
    tDisp.arrDisp[0].uDisp.tOSD.u32BmpHeight = img_overlay.height;
    tDisp.arrDisp[0].uDisp.tOSD.u32DstXoffset = 0;
    tDisp.arrDisp[0].uDisp.tOSD.u32DstYoffset = osd_pipe->m_output_type == po_vo_sipeed_maix3_screen ? 32 : 0;
    tDisp.arrDisp[0].uDisp.tOSD.u64PhyAddr = 0;
    tDisp.arrDisp[0].uDisp.tOSD.pBitmap = img_overlay.data;

    int ret = AX_IVPS_RGN_Update(osd_pipe->m_ivps_attr.n_osd_rgn_chn[0], &tDisp);
    if (0 != ret)
    {
        ALOGE("AX_IVPS_RGN_Update fail, ret=0x%x, hChnRgn=%d", ret, osd_pipe->m_ivps_attr.n_osd_rgn_chn[0]);
    }

    // CALC_FPS("g_m3axpi_display");
}

static py::list g_m3axpi_capture()
{
    if (!g_m3axpi.bRunState) return py::list();
    py::list return_img;
    pthread_mutex_lock(&g_m3axpi.g_capture_mutex);
    if (g_m3axpi.g_capture.rows)
    {
        return_img.append(g_m3axpi.g_capture.rows);
        return_img.append(g_m3axpi.g_capture.cols);
        return_img.append(3);
        py::bytes tmp((char *)g_m3axpi.g_capture.data, g_m3axpi.g_capture.rows * g_m3axpi.g_capture.cols * 3);
        return_img.append(tmp);
    }
    pthread_mutex_unlock(&g_m3axpi.g_capture_mutex);
    // CALC_FPS("g_m3axpi_capture");
    return return_img;
}

static py::dict g_m3axpi_forward()
{
    py::dict result;
    if (g_m3axpi.bRunState && g_m3axpi.bRunJoint && g_m3axpi.g_result_forward.size())
    {
        pthread_mutex_lock(&g_m3axpi.g_forward_mutex);
        axdl_results_t res = g_m3axpi.g_result_forward.front();
        g_m3axpi.g_result_forward.pop();
        pthread_mutex_unlock(&g_m3axpi.g_forward_mutex);

        if (res.mModelType == 0) {
            return result;
        }

        result["mModelType"] = res.mModelType;
        result["niFps"] = res.niFps;
        result["noFps"] = res.noFps;

        if (res.nObjSize)
        {
            result["nObjSize"] = res.nObjSize;
            result["mObjects"] = py::list();
            for (int i = 0, s = res.nObjSize; i < s; i++)
            {
                auto &obj = res.mObjects[i];
                py::dict obj_dict;
                obj_dict["label"] = obj.label;
                obj_dict["prob"] = obj.prob;
                obj_dict["objname"] = std::string(obj.objname);

                py::list bbox;
                bbox.append(obj.bbox.x);
                bbox.append(obj.bbox.y);
                bbox.append(obj.bbox.w);
                bbox.append(obj.bbox.h);
                obj_dict["bbox"] = bbox;

                obj_dict["bHasBoxVertices"] = obj.bHasBoxVertices;
                if (obj.bHasBoxVertices)
                {
                    py::list bbox_vertices;
                    for (int j = 0; j < 4; j++)
                    {
                        py::list point;
                        point.append(obj.bbox_vertices[j].x);
                        point.append(obj.bbox_vertices[j].y);
                        bbox_vertices.append(point);
                    }
                    obj_dict["bbox_vertices"] = bbox_vertices;
                }

                obj_dict["nLandmark"] = obj.nLandmark;
                if (obj.nLandmark)
                {
                    py::list landmark;
                    for (int j = 0; j < obj.nLandmark; j++)
                    {
                        py::list point;
                        point.append(obj.landmark[j].x);
                        point.append(obj.landmark[j].y);
                        landmark.append(point);
                    }
                    obj_dict["landmark"] = landmark;
                }

                obj_dict["bHasMask"] = obj.bHasMask;
                if (obj.bHasMask)
                {
                    py::bytes tmp((char *)obj.mYolov5Mask.data, obj.mYolov5Mask.h * obj.mYolov5Mask.w);
                    obj_dict["mYolov5Mask"] = tmp;
                }

                obj_dict["bHasFaceFeat"] = obj.bHasFaceFeat;
                if (obj.bHasFaceFeat)
                {
                    py::bytes tmp((char *)obj.mFaceFeat.data, obj.mFaceFeat.h * obj.mFaceFeat.w);
                    obj_dict["mFaceFeat"] = tmp;
                }

                result["mObjects"].cast<py::list>().append(obj_dict);
            }
        }

        if (res.bPPHumSeg)
        {
            result["bPPHumSeg"] = res.bPPHumSeg;
            py::bytes tmp((char *)res.mPPHumSeg.data, res.mPPHumSeg.h * res.mPPHumSeg.w);
            result["mPPHumSeg"] = tmp;
        }

        if (res.bYolopv2Mask)
        {
            result["bYolopv2Mask"] = res.bYolopv2Mask;
            py::bytes tmp((char *)res.mYolopv2seg.data, res.mYolopv2seg.h * res.mYolopv2seg.w);
            result["mYolopv2seg"] = tmp;

            tmp = py::bytes((char *)res.mYolopv2ll.data, res.mYolopv2ll.h * res.mYolopv2ll.w);
            result["mYolopv2ll"] = tmp;
        }

        if (res.nCrowdCount)
        {
            result["nCrowdCount"] = res.nCrowdCount;
            result["mCrowdCountPts"] = py::list();
            for (int i = 0, s = res.nCrowdCount; i < s; i++)
            {
                auto &obj = res.mCrowdCountPts[i];
                py::dict obj_dict;
                obj_dict["x"] = obj.x;
                obj_dict["y"] = obj.y;
                result["mCrowdCountPts"].cast<py::list>().append(obj_dict);
            }
        }
    }
    return result;
}

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

PYBIND11_MODULE(m3axpi, m) {

    m.def("load", &g_m3axpi_load, py::arg("CfgPath"));

    m.def("camera", &g_m3axpi_camera, py::arg("CameraWidth") = 640, py::arg("CameraHeight") = 360, py::arg("SysCase") = 0, py::arg("HdrMode") = 1, py::arg("FrameRate") = 30);

    m.def("display", &g_m3axpi_display, py::arg("img") = py::list());

    m.def("capture", &g_m3axpi_capture);

    m.def("forward", &g_m3axpi_forward);

    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
}
