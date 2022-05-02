/*!
* \file edge.cpp
* \brief LM非线性优化的残差参数实现类
*
*	将非线性优化的残差抽象为边数据结构
*	，通过求解边相对于顶点的雅可比矩阵
*	，进行LM的优化流程
*
* \author xinxin
* \version 1.0
* \date 2022-05-01
*/

#include "backend/vertex.h"
#include "backend/edge.h"
#include <iostream>

using namespace std;

namespace backend {

	unsigned long global_edge_id = 0;

	Edge::Edge(int residual_dimension, int num_verticies,
		const std::vector<std::string> &verticies_types) {
		/* 初始化残差&顶点类型&雅可比矩阵&顶点ID */
		residual_.resize(residual_dimension, 1);
		if (!verticies_types.empty()) {
			verticies_types_ = verticies_types;
		}
		jacobians_.resize(num_verticies);
		id_ = global_edge_id++;
		/* 初始化当前边对应的信息矩阵 */
		Eigen::MatrixXd information(residual_dimension, residual_dimension);
		information.setIdentity();
		information_ = information;
	}

	/* 边的实现中大量调用共享指针，因此不用显示释放成员变量 */
	Edge::~Edge() {}

	double Edge::Chi2() {
		/* 残差未预先乘以sqrt(information_) */
		return residual_.transpose() * information_ * residual_;
		/* 残差已预先乘以sqrt(information_) */
		//return residual_.transpose() * residual_; 
	}

	bool Edge::CheckValid() {
		/* 判断当前边顶点类型设置是否正确 */
		if (!verticies_types_.empty()) {
			for (size_t i = 0; i < verticies_.size(); ++i) {
				if (verticies_types_[i] != verticies_[i]->TypeInfo()) {
					std::cout << "Vertex Type Does Not Match, Should Be " << verticies_types_[i] <<
						", But Set To " << verticies_[i]->TypeInfo() << std::endl;
					return false;
				}
			}
		}
		/* 检查信息矩阵&残差矩阵&观测矩阵三者对应关系 */
		if (information_.rows() != information_.cols()) {
			std::cout << "Information Matrix Format Error, Reset It" << std::endl;
			return false;
		}
		if (information_.rows() != residual_.rows()) {
			std::cout << "Information Matrix Not Match Residual Matrix, Reset It" << std::endl;
			return false;
		}
		if (observation_.rows() != residual_.rows()) {
			std::cout << "Observation Matrix Not Match Residual Matrix, Reset It" << std::endl;
			return false;
		}
		/* 检查雅可比矩阵&残差是否满足对应关系 */
		for (size_t i = 0; i < jacobians_.size(); ++i) {
			if (jacobians_[i].rows() != residual_.rows()) {
				std::cout << "Jacobian Matrix Not Match Residual Matrix, Reset It" << std::endl;
				return false;
			}
			if (jacobians_[i].cols() != verticies_[i]->LocalDimension()) {
				std::cout << "Jacobian Matrix Not Match Vertical Matrix, Reset It" << std::endl;
				return false;
			}
		}
		/* 正常返回 */
		return true;
	}
}