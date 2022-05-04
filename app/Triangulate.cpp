#include <iostream>
#include <vector>
#include <random>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#define M_PI 3.14159265358979323846

struct Pose
{
	Pose(Eigen::Matrix3d R, Eigen::Vector3d t) :Rwc(R), qwc(R), twc(t) {};
	Eigen::Matrix3d Rwc;
	Eigen::Quaterniond qwc;
	Eigen::Vector3d twc;
	Eigen::Vector2d uv;
};

int main()
{
	int poseNums = 10;
    double radius = 8;
    double fx = 1.;
    double fy = 1.;
	/* 生成相机姿态 */
    std::vector<Pose> camera_pose;
    for(int n = 0; n < poseNums; ++n ) {
        double theta = n * 2 * M_PI / ( poseNums * 4);
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
		camera_pose.push_back(Pose(R, t));
    }
	/* 随机数生成一个三维特征点 */
    std::default_random_engine generator;
    std::uniform_real_distribution<double> xy_rand(-4, 4.0);
    std::uniform_real_distribution<double> z_rand(8., 10.);
    double tx = xy_rand(generator);
    double ty = xy_rand(generator);
    double tz = z_rand(generator);
    Eigen::Vector3d Pw(tx, ty, tz);
	/* 这个特征从第三帧开始观测 */
	int start_frame_id = 3;
	int end_frame_id = poseNums;
	for (int i = start_frame_id; i < end_frame_id; ++i) {
		Eigen::Matrix3d Rcw = camera_pose[i].Rwc.transpose();
		Eigen::Vector3d Pc = Rcw * Pw - camera_pose[i].twc;

		double x = Pc.x();
		double y = Pc.y();
		double z = Pc.z();

		camera_pose[i].uv = Eigen::Vector2d(x / z, y / z);
	}
	/* 遍历所有数据并完成三角化 */
	Eigen::Vector3d P_test;
	P_test.setZero();
	Eigen::MatrixXd D(2*(end_frame_id- start_frame_id),4);
	for (int i = start_frame_id; i < end_frame_id; ++i) {
		Eigen::MatrixXd P(4, 4);
		P.block(0, 0, 3, 3) = camera_pose[i].Rwc.transpose();
		P.block(0, 3, 3, 1) = -camera_pose[i].twc;
		
		D.block((i- start_frame_id) * 2, 0, 1, 4) = camera_pose[i].uv[0] * P.row(2) - P.row(0);
		D.block((i- start_frame_id) * 2 + 1, 0, 1, 4) = camera_pose[i].uv[1] * P.row(2) - P.row(1);
	}
	Eigen::MatrixXd D_res = D.transpose()*D;
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(D_res, Eigen::ComputeFullU | Eigen::ComputeFullV);
	auto res_U = svd.matrixU();
	auto res_V = svd.matrixV();
	P_test.block(0, 0, 3, 1) = ((res_U.rightCols(1)) / (res_U.rightCols(1)(3))).block(0, 0, 3, 1);
	/* 打印三角化点信息及真实值 */
	std::cout << "singular value:\n" << svd.singularValues() << std::endl;
	std::cout << "ground truth: \n" << Pw.transpose() << std::endl;
	std::cout << "your result: \n" << P_test.transpose() << std::endl;
	/* 正常退出 */
	system("pause");
	return 0;
}