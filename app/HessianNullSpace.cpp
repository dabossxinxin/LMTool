#include <iostream>
#include <vector>
#include <random>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#define M_PI 3.14159265358979323846

struct Pose
{
	Pose(Eigen::Matrix3d& R, Eigen::Vector3d& t) :Rwc(R), qwc(R), twc(t) {};
	Eigen::Matrix3d Rwc;
	Eigen::Quaterniond qwc;
	Eigen::Vector3d twc;
};

int main()
{
	int featureNums = 20;
	int poseNums = 10;
	int dim = poseNums * 6 + featureNums * 3;
	double fx = 1.;
	double fy = 1.;
	Eigen::MatrixXd H(dim, dim);
	H.setZero();

	std::vector<Pose> camera_pose;
	double radius = 8;
	for (int n = 0; n < poseNums; ++n) {
		double theta = n * 2 * M_PI / (poseNums * 4); // 1/4 Բ��
													  // �� z�� ��ת
		Eigen::Matrix3d R;
		R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
		Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
		camera_pose.push_back(Pose(R, t));
	}

	std::default_random_engine generator;
	std::vector<Eigen::Vector3d> points;
	for (int j = 0; j < featureNums; ++j)
	{
		std::uniform_real_distribution<double> xy_rand(-4., 4.0);
		std::uniform_real_distribution<double> z_rand(8., 10.);
		double tx = xy_rand(generator);
		double ty = xy_rand(generator);
		double tz = z_rand(generator);

		Eigen::Vector3d Pw(tx, ty, tz);
		points.push_back(Pw);

		for (int i = 0; i < poseNums; ++i) {
			Eigen::Matrix3d Rcw = camera_pose[i].Rwc.transpose();
			Eigen::Vector3d Pc = Rcw * Pw - camera_pose[i].twc;

			double x = Pc.x();
			double y = Pc.y();
			double z = Pc.z();
			double z_2 = z * z;
			/* �������زв�������������ſɱ� */
			Eigen::Matrix<double, 2, 3> jacobian_uv_Pc;
			jacobian_uv_Pc << fx / z, 0, -x * fx / z_2,
				0, fy / z, -y * fy / z_2;
			/* �������زв��������������ſɱ� */
			Eigen::Matrix<double, 2, 3> jacobian_Pj = jacobian_uv_Pc * Rcw;
			/* �������زв���������̬���ſɱ� */
			Eigen::Matrix<double, 2, 6> jacobian_Ti;
			jacobian_Ti << -x* y * fx / z_2, (1 + x*x / z_2)*fx, (-y / z)*fx, fx / z, 0, -x * fx / z_2,
				-(1 + y*y / z_2)*fy, (x*y / z_2) * fy, (x / z) * fy, 0, fy / z, -y * fy / z_2;
			/* ��в���������̬���ſɱ� */
			H.block(i * 6, i * 6, 6, 6) += jacobian_Ti.transpose() * jacobian_Ti;
			/* ��в���������������ſɱ� */
			H.block(j * 3 + 6 * poseNums, j * 3 + 6 * poseNums, 3, 3) += jacobian_Pj.transpose()*jacobian_Pj;
			/* �б�Խ���Ϣ�������� */
			H.block(i * 6, j * 3 + 6 * poseNums, 6, 3) += jacobian_Ti.transpose() * jacobian_Pj;
			H.block(j * 3 + 6 * poseNums, i * 6, 3, 6) += jacobian_Pj.transpose() * jacobian_Ti;
		}
	}
	/* SVD�ֽ�鿴�������ռ� */
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
	std::cout << svd.singularValues() << std::endl;
	/* �����˳� */
	system("pause");
	return 0;
}