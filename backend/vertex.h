#pragma once

/*!
* \file vertex.h
* \brief LM非线性优化的待优化参数声明类
*
*	将非线性优化的待优化参数抽象为顶点数据
*	数据结构，多种优化参数对应多组顶点
*
* \author xinxin
* \version 1.0
* \date 2022-05-01
*/

#include "backend/eigen_types.h"

/* 优化属于后端进行的任务，因此此处将命名空间定为backend */
namespace backend {
	/**
	* @brief 顶点类，对应一个parameter block，变量
	*		  值以VecX存储，需要在构造时指定维度
	*/
	class Vertex {
	public:
		/*! * @brief 保证向量空间内存对齐 */
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		/*!
		*  @brief  顶点构造函数
		*  @param[in]  num_dimension		顶点自身维度
		*  @param[in]  local_dimension		顶点本身参数化维度，为-1时与本身维度一致
		*/
		explicit Vertex(int num_dimension, int local_dimension = -1);
		/*!
		*  @brief  顶点析构函数
		*/
		virtual ~Vertex();
		/*!
		*  @brief  获取变量维度
		*  @return	int	待优化变量维度
		*/
		int Dimension() const;
		/*!
		*  @brief  获取变量本身参数化维度
		*  @return	int	待优化变量参数化维度
		*/
		int LocalDimension() const;
		/*!
		*  @brief  获取当前顶点的ID
		*  @return	int	当前顶点ID
		*/
		unsigned long Id() const {
			return id_;
		}
		/*!
		*  @brief  获取当前顶点的参数值
		*  @return	int	当前顶点参数值
		*/
		VecX Parameters() const {
			return parameters_;
		}
		/*!
		*  @brief  获取当前顶点的参数值的引用
		*  @return	int	当前顶点参数值的引用
		*/
		VecX &Parameters() {
			return parameters_;
		}
		/*!
		*  @brief  设置当前顶点的参数值的初始值
		*  @return	int	当前顶点参数值的初始值
		*/
		void SetParameters(const VecX &params) {
			parameters_ = params;
		}
		/*!
		*  @brief  顶点类中定义的广义加法，默认
		*		   为向量加法，可重定义
		*/
		virtual void Plus(const VecX &delta);
		/*!
		*  @brief  获取当前顶点类型名称
		*  @return	int	当前顶点的类型名称
		*/
		virtual std::string TypeInfo() const = 0;
		/*!
		*  @brief  获取当前顶点排序后ID
		*  @return	int	当前顶点排序后ID
		*/
		int OrderingId() const {
			return ordering_id_;
		}
		/*!
		*  @brief  设置当前顶点排序后ID
		*  @return	int	当前顶点排序后ID
		*/
		void SetOrderingId(unsigned long id) {
			ordering_id_ = id;
		};
		/*!
		*  @brief  是否固定当前参数值不参加优化
		*  @return	int	是否固定当前顶点的标志位
		*/
		void SetFixed(bool fixed = true) {
			fixed_ = fixed;
		}
		/*!
		*  @brief  获取是否固定当前参数值不参加优化
		*  @return	int	当前顶点是否优化的标志位
		*/
		bool IsFixed() const { return fixed_; }

	protected:
		/*! @brief	顶点参数值 */
		VecX			parameters_;
		/*! @brief	顶点自身参数化维度 */
		int				local_dimension_;
		/*! @brief	顶点的ID，自动生成 */
		unsigned long	id_;
		/*! @brief	优化问题中排序后的ID，用于寻找对应的雅可比块 */
		unsigned long	ordering_id_ = 0;
		/*! @brief	当前顶点是否固定的标志位 */
		bool fixed_ = false;
	};
}
