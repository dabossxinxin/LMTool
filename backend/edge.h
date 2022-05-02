#pragma once

/*!
* \file edge.h
* \brief LM非线性优化的残差参数声明类
*
*	将非线性优化的残差抽象为边数据结构
*	，通过求解边相对于顶点的雅可比矩阵
*	，进行LM的优化流程
*
* \author xinxin
* \version 1.0
* \date 2022-05-01
*/

#include <memory>
#include <string>
#include "backend/eigen_types.h"

namespace backend {
	/*! @brief	前置定义Vertex */
	class Vertex;
	/**
	* @brief 边类：负责计算优化残差
	* @detial 残差定义为：预测-观测，观测维度在构造函数中定义
	*		  代价函数是：残差*信息*残差，是一个数值，由后端求和后最小化
	*/
	class Edge {
	public:
		/*! * @brief 保证向量空间内存对齐 */
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		/*!
		*  @brief  边构造函数
		*  @param[in]  residual_dimension	残差维度
		*  @param[in]  num_verticies		顶点数量
		*  @param[in]  verticies_types		顶点类型名称，可以不给，不给的话check中不会检查
		*/
		explicit Edge(int residual_dimension, int num_verticies,
			const std::vector<std::string> &verticies_types = std::vector<std::string>());
		/*!
		*  @brief  边析构函数
		*/
		virtual ~Edge();
		/*!
		*  @brief  返回当前边ID
		*  @return	unsigned long	当前边ID
		*/
		unsigned long Id() const { 
			return id_; 
		}
		/*!
		*  @brief  设置当前边的顶点
		*  @param[in]	vertex	顶点数量
		*  @return		bool	是否设置成功
		*/
		bool AddVertex(std::shared_ptr<Vertex> vertex) {
			verticies_.emplace_back(vertex);
			return true;
		}
		/*!
		*  @brief  设置当前边的顶点
		*  @param[in]	vertex	顶点数量
		*  @return		bool	是否设置成功
		*/
		bool SetVertex(const std::vector<std::shared_ptr<Vertex>> &vertices) {
			verticies_ = vertices;
			return true;
		}
		/*!
		*  @brief  获取当前边索引为i的顶点
		*  @param[in]	int						顶点索引
		*  @return		std::shared_ptr<Vertex>	索引为i的索引
		*/
		std::shared_ptr<Vertex> GetVertex(int i) {
			return verticies_[i];
		}
		/*!
		*  @brief  获取当前边的所有顶点
		*  @return		std::shared_ptr<Vertex> 当前边所有顶点
		*/
		std::vector<std::shared_ptr<Vertex>> Verticies() const {
			return verticies_;
		}
		/*!
		*  @brief  获取当前边的顶点数量
		*  @return	size_t 当前边的顶点数量
		*/
		size_t NumVertices() const { 
			return verticies_.size(); 
		}
		/*!
		*  @brief  获取当前边类型信息，在子类中实现
		*  @return	std::string 当前边类型信息
		*/
		virtual std::string TypeInfo() const = 0;
		/*!
		*  @brief  计算当前边残差，由子类实现
		*/
		virtual void ComputeResidual() = 0;
		/*!
		*  @brief  计算当前边雅可比矩阵，由子类实现
		*/
		virtual void ComputeJacobians() = 0;
		/*!
		*  @brief  计算当前边平方误差，会乘以信息矩阵
		*/
		double Chi2();
		/*!
		*  @brief  计算当前边残差
		*/
		VecX Residual() const { 
			return residual_; 
		}
		/*!
		*  @brief  获取当前边相对于对应顶点的雅可比矩阵
		*  @return	std::vector<MatXX> 当前边与对应顶点的雅可比矩阵
		*/
		std::vector<MatXX> Jacobians() const {
			return jacobians_; 
		}
		/*!
		*  @brief  设置当前边对应的信息矩阵：information_ = sqrt_Omega = w
		*  @param[in]	information	当前边对应的信息矩阵
		*/
		void SetInformation(const MatXX &information) {
			information_ = information;
		}
		/*!
		*  @brief  获取当前边对应的信息矩阵
		*  @return	MatXX 当前边对应的信息矩阵
		*/
		MatXX Information() const {
			return information_;
		}
		/*!
		*  @brief  设置当前边对应的观测值
		*  @param[in]	observation	当前边对应的观测值 
		*/
		void SetObservation(const VecX &observation) {
			observation_ = observation;
		}
		/*!
		*  @brief  获取当前边对应的观测值
		*  @return	VecX	当前边对应的观测值
		*/
		VecX Observation() const { return observation_; }
		/*!
		*  @brief  检查边的信息是否全部设置
		*  @return	bool	当前边是否全部设置的标志位
		*/
		bool CheckValid();
		/*!
		*  @brief  获取当前边排序后的ID
		*  @return	int	当前边排序后的ID
		*/
		int OrderingId() const { 
			return ordering_id_;
		}
		/*!
		*  @brief  设置当前边排序后的ID
		*  @return	id	当前边排序后的ID
		*/
		void SetOrderingId(int id) {
			ordering_id_ = id; 
		};

	protected:
		/*! @brief	当前边的ID */
		unsigned long							id_;
		/*! @brief	当前边排序后的ID */
		int										ordering_id_;
		/*! @brief	当前边对应顶点的类型信息 */
		std::vector<std::string>				verticies_types_;
		/*! @brief	当前边对应的顶点信息 */
		std::vector<std::shared_ptr<Vertex>>	verticies_;
		/*! @brief	当前边对应的残差信息 */
		VecX									residual_;
		/*! @brief	当前边与对应顶点计算得到的雅可比信息 */
		std::vector<MatXX>						jacobians_;
		/*! @brief	当前边对应的信息矩阵 */
		MatXX									information_;
		/*! @brief	当前边对应的观测信息 */
		VecX									observation_;
	};

}
