#include <chrono>
#include "DWGProcess.h"

void DWGLaunch(std::string point_cloud_full_path, std::string out_mesh_path, std::string out_mesh_full_path,
    int input_normal, int grid_reso, int final_reso, int kw, int final_kw, int normal_kw, int max_depth, float sigma, int min_num, bool use_screened, bool use_radii, int com_As, float As_threshold, float offset_1, float offset_2, int use_leaves_generate,
    int total_iter, int test_iter, int output_mode,
    float learning_rate_Ai, float learning_rate_Ai_2, float learning_rate_Ci, int step_iteration, float diric_scale, int diric_reso, float ai_reg, float ai_reg_2, float ci_reg_c, float ci_reg_a,
    float ai_rad, float ci_rad, float weight_scale, bool pre_process,
    int before_ci_max_loop,
    int ci_max_loop_iteration,
    int max_loop,
    float ai_target_start,
    float ai_target_mid,

    float ci_target,
    float ai_smooth_rad, bool ai_adjust,

    float ci_in_thres, float ci_out_thres,
    float ci_final_in_thres, float ci_final_out_thres,
    int ci_level,
    float ci_var_thres,
    float ai_var_thres_0,
    float ai_var_thres_1,
    int DWG_extract_mode,
    int DWG_initial_mode,
    float inside_x,
    float inside_y,
    float inside_z)
{
    clock_t start;
    start = clock();

    std::chrono::high_resolution_clock::time_point start_time_s = std::chrono::high_resolution_clock::now();

    test3D::DataprePointCloud(point_cloud_full_path, max_depth, input_normal, com_As, pre_process, start, start_time_s);

    test3D::TurboIterative(start, start_time_s, out_mesh_path, out_mesh_full_path, input_normal, grid_reso, final_reso, kw, final_kw, normal_kw, max_depth, sigma, min_num, use_screened, use_radii, com_As, As_threshold, offset_1, offset_2, use_leaves_generate, total_iter, test_iter, output_mode,
        learning_rate_Ai, learning_rate_Ai_2, learning_rate_Ci, step_iteration, diric_scale, diric_reso, ai_reg, ai_reg_2, ci_reg_c, ci_reg_a,
        ai_rad, ci_rad, weight_scale,
        before_ci_max_loop,
        ci_max_loop_iteration,
        max_loop,
        ai_target_start,
        ai_target_mid,

        ci_target,
        ai_smooth_rad, ai_adjust,

        ci_in_thres, ci_out_thres,
        ci_final_in_thres, ci_final_out_thres,
        ci_level,
        ci_var_thres,
        ai_var_thres_0,
        ai_var_thres_1,
        DWG_extract_mode,
        DWG_initial_mode,
        inside_x,
        inside_y,
        inside_z);
}

int main(int argc, char* argv[])
{
    std::string point_cloud_path = "E:/dwg++/model/";
    std::string file_name = "Fig1_yoga_teaser_input.xyz";
    std::string out_mesh_path = "E:/dwg++/mesh/0523";
    int input_normal = 0;
    int grid_reso = 260;
    int final_reso = 260;
    int kw = 6;
    int final_kw = 6;
    int normal_kw = 10;
    int max_depth = 10;
    int total_iter = 60;
    float sigma = 0;
    int output_mode = 3;
    int min_num = 0;
    int use_leaves_generate = 0;
    bool use_screened = 1;
    bool use_radii = 1;
    int com_As = 1;
    float As_threshold = 0.0;
    float offset_1 = 0.05;
    float offset_2 = 0.01;
    int test_iter = 99999;

    float ai_lr = 1e-6f;
    float ai_lr_2 = 2e-7f;
    float ci_lr = 5e-3f;
    float diric_scale = 5e-2f;
    float diric_reso = 64;
    float ai_reg = 10.0f;
    float ai_reg_2 = 1.0f;
    float ci_reg_c = 3e-4f;
    float ci_reg_a = 0.5f;
    float ci_radius = 0.06f;
    float ai_radius = 0.04f;
    int opt_step = 150;
    float weight_scale = 1.0f;
    bool pre_process = false;

    int before_ci_max_loop = 5;
    int ci_max_loop_iteration = 300;
    float ai_target_start = 60.0f  ;
    float ai_target_mid = 10.0f  ;

    float ci_target = 1.0f;
    int ci_level = 128;
    float ai_var_thres_0 = 0.15f;
    float ai_var_thres_1 = 0.15f;
    float ci_var_thres = 0.03f;
    float inside_x = 0.0f;
    float inside_y = 0.0f;
    float inside_z = 0.0f;
    int DWG_extract_mode = 0;
    int DWG_initial_mode = 0;
    int max_loop = 3;
    float ai_smooth_rad = 0.02f;
    bool ai_adjust = true;

    float ci_in_thres = 0.05f;
    float ci_out_thres = 0.05f;
    float ci_final_in_thres = 0.0f;
    float ci_final_out_thres = 0.0f;

    for (int i = 1; i < argc; i += 2)
    {
        if (std::strcmp(argv[i], "--in_path") == 0)
        {
            point_cloud_path = argv[i + 1];
        }
        else if (std::strcmp(argv[i], "--in_name") == 0)
        {
            file_name = argv[i + 1];
        }
        else if (std::strcmp(argv[i], "--out_path") == 0)
        {
            out_mesh_path = argv[i + 1];
        }
        else if (std::strcmp(argv[i], "--in_normal") == 0)
        {
            input_normal = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--grid_reso") == 0)
        {
            grid_reso = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--final_reso") == 0)
        {
            final_reso = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--max_depth") == 0)
        {
            max_depth = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--kw") == 0)
        {
            kw = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--final_kw") == 0)
        {
            final_kw = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--normal_kw") == 0)
        {
            normal_kw = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--lambda") == 0)
        {
            sigma = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--min_num") == 0)
        {
            min_num = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--total_iter") == 0)
        {
            total_iter = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--output_mode") == 0)
        {
            output_mode = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--use_leaves_generate") == 0)
        {
            use_leaves_generate = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--use_screened") == 0)
        {
            use_screened = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--use_radii") == 0)
        {
            use_radii = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--com_As") == 0)
        {
            com_As = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--As_thres") == 0)
        {
            As_threshold = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--offset_1") == 0)
        {
            offset_1 = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--offset_2") == 0)
        {
            offset_2 = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--test_iter") == 0)
        {
            test_iter = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--ai_lr") == 0)
        {
            ai_lr = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--ai_lr_2") == 0)
        {
            ai_lr_2 = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--ci_lr") == 0)
        {
            ci_lr = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--di_s") == 0)
        {
            diric_scale = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--di_res") == 0)
        {
            diric_reso = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--ai_reg") == 0)
        {
            ai_reg = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--ai_reg_2") == 0)
        {
            ai_reg_2 = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--ci_reg_c") == 0)
        {
            ci_reg_c = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--ci_reg_a") == 0)
        {
            ci_reg_a = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--ai_rad") == 0)
        {
            ai_radius = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--ci_rad") == 0)
        {
            ci_radius = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--opt_step") == 0)
        {
            opt_step = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--weight_s") == 0)
        {
            weight_scale = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--pre_p") == 0)
        {
            pre_process = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--before_ci_max_loop") == 0)
        {
            before_ci_max_loop = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--ci_max_loop_iter") == 0)
        {
            ci_max_loop_iteration = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--ai_target_start") == 0)
        {
            ai_target_start = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--ai_target_mid") == 0)
        {
            ai_target_mid = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--ci_target") == 0)
        {
            ci_target = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--ci_level") == 0)
        {
            ci_level = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--ai_var_thres_0") == 0)
        {
            ai_var_thres_0 = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--ai_var_thres_1") == 0)
        {
            ai_var_thres_1 = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--ci_var_thres") == 0)
        {
            ci_var_thres = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--DWG_extract_mode") == 0)
        {
            DWG_extract_mode = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--DWG_initial_mode") == 0)
        {
            DWG_initial_mode = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--max_loop") == 0)
        {
            max_loop = std::stoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--ai_smooth_rad") == 0)
        {
            ai_smooth_rad = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--ai_adjust") == 0)
        {
            ai_adjust = std::stoi(argv[i + 1]) != 0;
        }
        else if (std::strcmp(argv[i], "--ci_in_thres") == 0)
        {
            ci_in_thres = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--ci_out_thres") == 0)
        {
            ci_out_thres = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--ci_final_in_thres") == 0)
        {
            ci_final_in_thres = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--ci_final_out_thres") == 0)
        {
            ci_final_out_thres = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--inside_x") == 0)
        {
            inside_x = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--inside_y") == 0)
        {
            inside_y = std::stof(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--inside_z") == 0)
        {
            inside_z = std::stof(argv[i + 1]);
        }
    }

    std::string point_cloud_full_path = point_cloud_path + "/" + file_name;
    std::string pure_name = file_name.substr(0, file_name.length() - 4);
    std::string mesh_full_name = pure_name + "_md_" + std::to_string(max_depth) + "_gr_" + std::to_string(grid_reso) + "_iter_" + std::to_string(total_iter);

    std::string out_mesh_full_path = out_mesh_path + "/" + mesh_full_name;
    std::string out_mesh_path_dwg = out_mesh_path + "/";

    if (((com_As == 1) || (com_As == 4) || (com_As == 5)) && input_normal == 0)
    {
        input_normal = 2;
    }

    DWGLaunch(point_cloud_full_path, out_mesh_path_dwg, out_mesh_full_path,
        input_normal, grid_reso, final_reso, kw, final_kw, normal_kw, max_depth, sigma, min_num, use_screened, use_radii, com_As, As_threshold, offset_1, offset_2, use_leaves_generate, total_iter, test_iter, output_mode,
        ai_lr, ai_lr_2, ci_lr, opt_step, diric_scale, diric_reso, ai_reg, ai_reg_2, ci_reg_c, ci_reg_a,
        ai_radius, ci_radius, weight_scale, pre_process,
        before_ci_max_loop,
        ci_max_loop_iteration,
        max_loop,
        ai_target_start,
        ai_target_mid,

        ci_target,
        ai_smooth_rad, ai_adjust,

        ci_in_thres, ci_out_thres,
        ci_final_in_thres, ci_final_out_thres,
        ci_level,
        ci_var_thres,
        ai_var_thres_0,
        ai_var_thres_1,
        DWG_extract_mode,
        DWG_initial_mode,
        inside_x,
        inside_y,
        inside_z);

    return 0;
}
