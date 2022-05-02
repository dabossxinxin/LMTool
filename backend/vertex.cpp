/*!
* \file vertex.cpp
* \brief LM非线性优化的待优化参数实现类
*
*	将非线性优化的待优化参数抽象为顶点数据
*	数据结构，多种优化参数对应多组顶点
*
* \author xinxin
* \version 1.0
* \date 2022-05-01
*/

#include "backend/vertex.h"
#include <iostream>

namespace backend {

	unsigned long global_vertex_id = 0;

	Vertex::Vertex(int num_dimension, int local_dimension) {
		parameters_.resize(num_dimension, 1);
		local_dimension_ = local_dimension > 0 ? local_dimension : num_dimension;
		id_ = global_vertex_id++;
	}

	Vertex::~Vertex() {}

	int Vertex::Dimension() const {
		return parameters_.rows();
	}

	int Vertex::LocalDimension() const {
		return local_dimension_;
	}

	void Vertex::Plus(const VecX &delta) {
		parameters_ += delta;
	}
}