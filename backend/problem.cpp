/*!
* \file problem.cpp
* \brief LM非线性优化问题实现类
*
*	构造非线性优化问题，并且将顶点与边
*	加入到优化问题中，求解优化后顶点的值
*
* \author xinxin
* \version 1.0
* \date 2022-05-01
*/

#include <iostream>
#include <fstream>
#include <Eigen/Dense>

#include "backend/problem.h"
#include "utils/tic_toc.h"

#ifdef USE_OPENMP
	#include <omp.h>
#endif

using namespace std;

namespace backend {

	void Problem::LogoutVectorSize() {
		std::cout << "Verticies: " << verticies_.size()
			<< "\tEdges: " << edges_.size() << std::endl;
	}

	Problem::Problem(ProblemType problemType) :
		problemType_(problemType) {
		//LogoutVectorSize();
		verticies_marg_.clear();
	}

	Problem::~Problem() {}

	bool Problem::AddVertex(std::shared_ptr<Vertex> vertex) {
		/* 在优化问题中添加顶点 */
		if (verticies_.find(vertex->Id()) != verticies_.end()) {
			std::cerr << "Vertex " << vertex->Id() << "Has Been Added Before" << std::endl;
			return false;
		}
		else {
			verticies_.insert(std::pair<unsigned long, shared_ptr<Vertex>>(vertex->Id(), vertex));
		}
		/* 正常返回 */
		return true;
	}

	bool Problem::AddEdge(shared_ptr<Edge> edge) {
		/* 在优化问题中添加边 */
		if (edges_.find(edge->Id()) == edges_.end()) {
			edges_.insert(pair<ulong, std::shared_ptr<Edge>>(edge->Id(), edge));
		}
		else {
			std::cerr << "Edge " << edge->Id() << "Has Been Added Before" << std::endl;
			return false;
		}
		/* 更新由Vertex查询Edge的数据结构 */
		for (auto &vertex : edge->Verticies()) {
			vertexToEdge_.insert(pair<ulong, shared_ptr<Edge>>(vertex->Id(), edge));
		}
		/* 正常返回 */
		return true;
	}

	bool Problem::Solve(int iterations) {
		/* 非法情况的判断与处理 */
		if (edges_.size() == 0 || verticies_.size() == 0) {
			std::cerr << "\nCannot Solve Problem Without Edges Or Verticies" << std::endl;
			return false;
		}
		/* 计时类 */
		TicToc t_solve;
		/* 统计优化变量维度，为构建H做准备 */
		this->SetOrdering();
		/* 遍历edge，构建H = Jt*J矩阵 */
		this->MakeHessian();
		/* 优化问题初始化 */
		this->ComputeLambdaInitLM();
		/* 初始化迭代次数&迭代停止标志 */
		int iter = 0;
		bool stop = false;
		/* 记录数据Lambda与迭代次数 */
		std::ofstream outFile;
		outFile.open("./lambda.txt");
		/* 开始进行非线性优化迭代 */
		while (!stop && (iter < iterations)) {
			std::cout << "Iter: " << iter
				<< ", Chi= " << currentChi_
				<< ", Lambda= " << currentLambda_
				<< ", DeltaX= " << delta_x_.squaredNorm()
				<< std::endl;
			outFile << iter << " " << currentLambda_ << std::endl;
			/* 当前迭代是否成功的标志位 */
			bool oneStepSuccess = false;
			/* 当前迭代连续失败的次数 */
			int false_cnt = 0;
			/* 不断尝试新的Lambda,知道跌倒成功 */
			while (!oneStepSuccess) {
				/* Lambda影响添加到海森矩阵中 */
				this->AddLambdatoHessianLM();
				/* 求解当前迭代的delta_x */
				this->SolveLinearSystem();
				/* 将Lambda的影响从海森矩阵中消除 */
				this->RemoveLambdaHessianLM();
				/* 优化退出条件1：当delta_x很小时，可认为迭代收敛 */
				if (delta_x_.squaredNorm() <= 1e-6 || false_cnt > 10) {
					stop = true;
					break;
				}
				/* 更新状态量 */
				this->UpdateStates();
				/* 判断当前迭代是否使残差下降 */
				oneStepSuccess = IsGoodStepInLM();
				// 后续处理，
				if (oneStepSuccess) {
					/* 在新的线性化点构建海森矩阵 */
					this->MakeHessian();
					/* 计算残差最大值 */
					double b_max = 0.0;
					for (int i = 0; i < b_.size(); ++i) {
					    b_max = max(fabs(b_(i)), b_max);
					}
					/* 优化退出条件2：如果残差b_max已经很小则退出 */
					stop = (b_max <= 1e-12);
					false_cnt = 0;
				} else {
					/* 当前迭代不成功：回退 */
					false_cnt++;
					this->RollbackStates();
				}
			}
			iter++;
			/* 优化退出条件3：currentChi_与第一次相比下降了1e6倍 */
			if ((currentChi_) <= stopThresholdLM_) {
				stop = true;
			}
		}
		std::cout << "Problem Solve Cost: " << t_solve.toc() << " ms" << std::endl;
		std::cout << "Make Hessian Cost: " << t_hessian_cost_ << " ms" << std::endl;
		return true;
	}

	void Problem::SetOrdering() {
		/* 每次重新统计维度信息 */
		ordering_poses_ = 0;
		ordering_generic_ = 0;
		ordering_landmarks_ = 0;
		/* 通用优化问题中，统计所有优化变量的总维度 */
		for (auto vertex : verticies_) {
			ordering_generic_ += vertex.second->LocalDimension();
		}
	}

	/* TODO:如何加速海森矩阵的构造 */
	/* 方案1：使用多线程加速 */
	/* 方案2：使用CUDA硬件加速 */
	void Problem::MakeHessian() {
		TicToc t_h;
		/* 初始化海森矩阵 */
		ulong size = ordering_generic_;
		MatXX H(MatXX::Zero(size, size));
		VecX b(VecX::Zero(size));
		/* 遍历每个残差项，并计算雅可比 */
		for (auto &edge : edges_) {
			/* 计算当前边的残差&雅可比 */
			edge.second->ComputeResidual();
			edge.second->ComputeJacobians();
			/* 获取当前边的雅可比矩阵以及顶点 */
			auto jacobians = edge.second->Jacobians();
			auto verticies = edge.second->Verticies();
			assert(jacobians.size() == verticies.size());
			if (jacobians.size() != verticies.size()) {
				std::cout << "Jacobian's Size Not Match Verticies's Size" << std::endl;
				return;
			}
			/* 计算单个残差构造的海森矩阵 */
			for (size_t i = 0; i < verticies.size(); ++i) {
				/* 若顶点固定，则顶点对应的雅可比不需要更新 */
				auto v_i = verticies[i];
				if (v_i->IsFixed()) continue;

				auto jacobian_i = jacobians[i];
				ulong index_i = v_i->OrderingId();
				ulong dim_i = v_i->LocalDimension();

				MatXX JtW = jacobian_i.transpose() * edge.second->Information();
				for (size_t j = i; j < verticies.size(); ++j) {
					/* 若顶点固定，则顶点对应的雅可比不需要更新 */
					auto v_j = verticies[j];
					if (v_j->IsFixed()) continue;

					auto jacobian_j = jacobians[j];
					ulong index_j = v_j->OrderingId();
					ulong dim_j = v_j->LocalDimension();

					assert(v_j->OrderingId() != -1);
					MatXX hessian = JtW * jacobian_j;
					/* 所有的信息矩阵叠加起来 */
					H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
					if (j != i) {
						/* 海森矩阵是对称的矩阵 */
						H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
					}
				}
				b.segment(index_i, dim_i).noalias() -= JtW * edge.second->Residual();
			}
		}
		/* TODO：如何修改掉赋值操作 */
		b_ = b;
		Hessian_ = H;
		t_hessian_cost_ += t_h.toc();
		/* 初始化状态增量 */
		delta_x_ = VecX::Zero(size);
	}

	void Problem::SolveLinearSystem() {
		/* 直接使用Eigen求逆 */
		delta_x_ = Hessian_.inverse() * b_;
		/* 使用Cholexy分解法求解线性方程 */
		// delta_x_ = H.ldlt().solve(b_);
		/* 使用共轭梯度下降法 */
		//delta_x_ = this->PCGSolver(Hessian_, b_, 5);
	}

	void Problem::UpdateStates() {
		/* 为所有顶点附加一个增量 */
		for (auto vertex : verticies_) {
			ulong idx = vertex.second->OrderingId();
			ulong dim = vertex.second->LocalDimension();
			VecX delta = delta_x_.segment(idx, dim);
			vertex.second->Plus(delta);
		}
	}

	void Problem::RollbackStates() {
		for (auto vertex : verticies_) {
			ulong idx = vertex.second->OrderingId();
			ulong dim = vertex.second->LocalDimension();
			VecX delta = delta_x_.segment(idx, dim);
			/* 上一步的更新量使得残差变大，因此此时消除上一步更新量的影响 */
			vertex.second->Plus(-delta);
		}
	}

	void Problem::ComputeLambdaInitLM() {
		/* 初始化迭代参数 */
		ni_ = 2.;
		currentLambda_ = -1.;
		currentChi_ = 0.0;
		// TODO:: robust cost chi2
		for (auto edge : edges_) {
			currentChi_ += edge.second->Chi2();
		}
		if (err_prior_.rows() > 0) {
			currentChi_ += err_prior_.norm();
		}
		/* 设置迭代的中止阈值 */
		stopThresholdLM_ = 1e-8 * currentChi_;
		/* 计算海森矩阵最大对角元素 */
		double maxDiagonal = 0;
		ulong size = Hessian_.cols();
		if (Hessian_.rows() != Hessian_.cols()) {
			std::cerr << "Hessian Is Not a Square Matrix" << std::endl;
			return;
		}
		for (ulong i = 0; i < size; ++i) {
			maxDiagonal = std::max(fabs(Hessian_(i, i)), maxDiagonal);
		}
		double tau = 1e-5;
		currentLambda_ = tau * maxDiagonal;
	}

	void Problem::AddLambdatoHessianLM() {
		ulong size = Hessian_.cols();
		if (Hessian_.rows() != Hessian_.cols()) {
			std::cerr << "Hessian Is Not a Square Matrix" << std::endl;
			return;
		}
		for (ulong i = 0; i < size; ++i) {
			Hessian_(i, i) += currentLambda_;
		}
	}

	void Problem::RemoveLambdaHessianLM() {
		ulong size = Hessian_.cols();
		if (Hessian_.rows() != Hessian_.cols()) {
			std::cerr << "Hessian Is Not a Square Matrix" << std::endl;
			return;
		}
		for (ulong i = 0; i < size; ++i) {
			Hessian_(i, i) -= currentLambda_;
		}
	}

	bool Problem::IsGoodStepInLM() {
		/* rho计算时的分母，为保证分母不为0，计算时加上1e-3 */
		double scale = 0;
		scale = 0.5*delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
		scale += 1e-3;
		/* 当前迭代后的残差总和 */
		double tempChi = 0.0;
		for (auto edge : edges_) {
			edge.second->ComputeResidual();
			tempChi += edge.second->Chi2();
		}
		/* 计算rho的值，从而确定后续迭代操作步骤 */
		double rho = (currentChi_ - tempChi) / scale;
		if (rho > 0 && isfinite(tempChi))
		{
			double alpha = 1. - pow((2 * rho - 1), 3);
			alpha = std::min(alpha, 2. / 3.);
			double scaleFactor = (std::max)(1. / 3., alpha);
			currentLambda_ *= scaleFactor;
			ni_ = 2;
			currentChi_ = tempChi;
			return true;
		}
		else {
			currentLambda_ *= ni_;
			ni_ *= 2;
			return false;
		}
	}

	/* Conjugate gradient with perconditioning(PCG) */
	VecX Problem::PCGSolver(MatXX &A, VecX &b, int maxIter) {
		if (A.rows() != A.cols()) {
			std::cerr << "PCG Solver ERROR: A Is Not a Square Matrix" << std::endl;
			return VecX::Zero(A.rows());
		}
		int rows = b.rows();
		int n = maxIter < 0 ? rows : maxIter;
		VecX x(VecX::Zero(rows));
		MatXX M_inv = A.diagonal().asDiagonal().inverse();
		VecX r0(b);  // initial r = b - A*0 = b
		VecX z0 = M_inv * r0;
		VecX p(z0);
		VecX w = A * p;
		double r0z0 = r0.dot(z0);
		double alpha = r0z0 / p.dot(w);
		VecX r1 = r0 - alpha * w;
		int i = 0;
		double threshold = 1e-6 * r0.norm();
		while (r1.norm() > threshold && i < n) {
			i++;
			VecX z1 = M_inv * r1;
			double r1z1 = r1.dot(z1);
			double belta = r1z1 / r0z0;
			z0 = z1;
			r0z0 = r1z1;
			r0 = r1;
			p = belta * p + z1;
			w = A * p;
			alpha = r1z1 / p.dot(w);
			x += alpha * p;
			r1 -= alpha * w;
		}
		return x;
	}
}






