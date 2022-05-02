#pragma once

/*!
* \file problem.h
* \brief LM非线性优化问题声明类
*
*	构造非线性优化问题，并且将顶点与边
*	加入到优化问题中，求解优化后顶点的值
*
* \author xinxin
* \version 1.0
* \date 2022-05-01
*/

#include <map>
#include <memory>
#include <unordered_map>

#include "backend/eigen_types.h"
#include "backend/edge.h"
#include "backend/vertex.h"

namespace backend {
	
	typedef unsigned long ulong;

	class Problem {
	public:
		/*! * @brief 保证向量空间内存对齐 */
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		/**
		* 问题的类型
		* SLAM问题还是通用的问题
		*
		* 如果是SLAM问题那么pose和landmark是区分开的，Hessian以稀疏方式存储
		* SLAM问题只接受一些特定的Vertex和Edge
		* 如果是通用问题那么hessian是稠密的，除非用户设定某些vertex为marginalized
		*/
		enum ProblemType {
			SLAM_PROBLEM,
			GENERIC_PROBLEM
		};
		
		typedef std::map<unsigned long, std::shared_ptr<Vertex>>				HashVertex;
		typedef std::unordered_map<unsigned long, std::shared_ptr<Edge>>		HashEdge;
		typedef std::unordered_multimap<unsigned long, std::shared_ptr<Edge>>	HashVertexIdToEdge;
		/*!
		*  @brief 优化问题构造函数
		*  @param[in]  problemType	优化问题类型
		*/
		Problem(ProblemType problemType);
		/*!
		*  @brief 优化问题析构函数
		*/
		~Problem();
		/*!
		*  @brief 向优化问题中添加待优化顶点
		*  @param[in]	vertex	待优化顶点
		*  @retuan		bool	是否成功添加顶点
		*/
		bool AddVertex(std::shared_ptr<Vertex> vertex);
		/*!
		*  @brief 移除优化问题中特定待优化顶点
		*  @param[in]  vertex	待优化顶点
		*  @retuan		bool	是否移除顶点
		*/
		bool RemoveVertex(std::shared_ptr<Vertex> vertex);
		/*!
		*  @brief 向优化问题中添加边
		*  @param[in]	vertex	待添加边
		*  @retuan		bool	是否成功添加边
		*/
		bool AddEdge(std::shared_ptr<Edge> edge);
		/*!
		*  @brief 移除优化问题中特定边
		*  @param[in]	vertex	边
		*  @retuan		bool	是否成功移除边
		*/
		bool RemoveEdge(std::shared_ptr<Edge> edge);
		/*!
		*  @brief 获取优化问题中被判定为外点的边，方便前端操作
		*  @param[in]	outlier_edges	外点边
		*/
		void GetOutlierEdges(std::vector<std::shared_ptr<Edge>> &outlier_edges);
		/*!
		*  @brief 求解当前优化问题
		*  @param[in]	iterations	非线性优化迭代次数
		*  @return		bool		非线性优化问题是否成功收敛
		*/
		bool Solve(int iterations);
		/*!
		*  @brief SLAM问题中边缘化一帧图像以及对应的特征
		*  @param[in]	frameVertex			边缘化的帧位置&姿态顶点
		*  @return		landmarkVerticies	边缘化的3D点顶点
		*/
		bool Marginalize(std::shared_ptr<Vertex> frameVertex,
			const std::vector<std::shared_ptr<Vertex>> &landmarkVerticies);
		/*!
		*  @brief SLAM问题中边缘化一帧图像
		*  @param[in]	frameVertex			边缘化的帧位置&姿态顶点
		*/
		bool Marginalize(const std::shared_ptr<Vertex> frameVertex);
		/*!
		*  @brief test compute prior
		*/
		void TestComputePrior();

	private:
		/*!
		*  @brief Solve函数的实现：求解通用问题
		*/
		bool SolveGenericProblem(int iterations);

		/*!
		*  @brief Solve函数的实现：求解SLAM问题
		*/
		bool SolveSLAMProblem(int iterations);
		/*!
		*  @brief 设置各个顶点的排序后ID
		*/
		void SetOrdering();
		/*!
		*  @brief SLAM问题中设置各个顶点的排序后ID
		*/
		void AddOrderingSLAM(std::shared_ptr<Vertex> v);
		/*!
		*  @brief 构造海森矩阵
		*/
		void MakeHessian();
		/*!
		*  @brief 舒尔补求解SBA问题
		*/
		void SchurSBA();
		/*!
		*  @brief 求解线性方程
		*/
		void SolveLinearSystem();
		/*!
		*  @brief 更新状态变量
		*/
		void UpdateStates();
		/*!
		*  @brief 更新残差反而变大时，此时退回上一步重新迭代
		*/
		void RollbackStates();
		/*!
		*  @brief 边缘化时计算先验部分
		*/
		void ComputePrior();
		/*!
		*  @brief 判断顶点是否为Pose顶点
		*/
		bool IsPoseVertex(std::shared_ptr<Vertex> v);
		/*!
		*  @brief 判断顶点是否为LandMark顶点
		*/
		bool IsLandmarkVertex(std::shared_ptr<Vertex> v);
		/*!
		*  @brief 新增顶点后，需调整几个海森矩阵的大小
		*/
		void ResizePoseHessiansWhenAddingPose(std::shared_ptr<Vertex> v);
		/*!
		*  @brief 检查顶点&边的顺序是否一一对应
		*/
		bool CheckOrdering();
		/*!
		*  @brief 调试用
		*/
		void LogoutVectorSize();
		/*!
		*  @brief 获取某个顶点链接到的边
		*/
		std::vector<std::shared_ptr<Edge>> GetConnectedEdges(std::shared_ptr<Vertex> vertex);
		/*!
		*  @brief 计算LM问题中的初始Lambda值
		*/
		void ComputeLambdaInitLM();
		/*!
		*  @brief 在海森矩阵对角线上加上Lambda
		*/
		void AddLambdatoHessianLM();
		/*!
		*  @brief 在海森矩阵对角线上减去Lambda
		*/
		void RemoveLambdaHessianLM();
		/*!
		*  @brief 用于判断当前迭代是否满足要求，指导Lambda的缩放
		*/
		bool IsGoodStepInLM();
		/*!
		*  @brief PCG迭代线性求解器
		*/
		VecX PCGSolver(MatXX &A, VecX &b, int maxIter = -1);

	private:
		double currentLambda_;		// 当前Lambda
		double currentChi_;			// 当前残差的平方和
		double stopThresholdLM_;    // LM 迭代退出阈值条件
		double ni_;                 // 控制Lambda缩放大小

		ProblemType problemType_;	// 优化问题类型

		MatXX Hessian_;				// 优化问题Jt*J
		VecX b_;					// 优化问题-Jt*f
		VecX delta_x_;				// 优化问题步长

		MatXX H_prior_;				// Jt*J矩阵先验
		VecX b_prior_;				// -Jt*f矩阵先验
		MatXX Jt_prior_inv_;		// Jt矩阵先验的逆
		VecX err_prior_;			// 残差先验

		MatXX H_pp_schur_;			// SBA的Pose部分
		VecX b_pp_schur_;			// SBA的Pose部分

		MatXX H_pp_;				// Jt*J的Pose部分
		VecX b_pp_;					// -Jt*f的Pose部分
		MatXX H_ll_;				// Jt*J的LandMark部分
		VecX b_ll_;					// -Jt*f的LandMark部分

		HashVertex verticies_;				// 优化问题的所有顶点
		HashEdge edges_;					// 优化问题的所有边
		HashVertexIdToEdge vertexToEdge_;	// 由Vertex的Id查询Edge 

		ulong ordering_poses_ = 0;			// Pose Order相关
		ulong ordering_landmarks_ = 0;		// LandMark Order相关
		ulong ordering_generic_ = 0;		// 通用问题Order相关

		std::map<unsigned long, std::shared_ptr<Vertex>>	idx_pose_vertices_;        // Order后的Pose顶点
		std::map<unsigned long, std::shared_ptr<Vertex>>	idx_landmark_vertices_;    // Order后的LandMark顶点
		HashVertex											verticies_marg_;		   // 需边缘化的Order后顶点

		bool bDebug = false;			// 是否调试
		double t_hessian_cost_ = 0.0;	// 海森矩阵计算时间
		double t_PCGsovle_cost_ = 0.0;	// PCG迭代线性求解时间
	};
}
