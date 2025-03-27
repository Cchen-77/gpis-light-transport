#ifndef GPLEVELCROSSINGKDTREE_HPP_
#define GPLEVELCROSSINGKDTREE_HPP_
#include "core/math/Vec.hpp"
#include "core/math/Ray.hpp"
#include "core/math/Box.hpp"
#include "core/math/GaussianProcess.hpp"

#include<tuple>

#include "core/io/JsonUtils.hpp"
#include "core/io/JsonSerializable.hpp"

#include <boost/math/distributions/normal.hpp>
namespace Tungsten {
class GPLevelCrossingKdTree {
public:
	struct Node {
		Node* children[2];
		int splitAxis;
		Box3f bounding;

		bool isLeaf() const {
			return children[0] == nullptr && children[1] == nullptr;
		}

		rapidjson::Value toJson(rapidjson::Document::AllocatorType& allocator) {
			auto result =  JsonObject{
				allocator,
				"splitAxis",splitAxis,
				"min",JsonUtils::toJson(bounding.min(),allocator),
				"max",JsonUtils::toJson(bounding.max(),allocator),
			};
			if (children[0]) {
				result.add("child0", children[0]->toJson(allocator));
			}
			if (children[1]) {
				result.add("child1", children[1]->toJson(allocator));
			}
			return result;
		}

		static Node* FromJson(JsonPtr json) {
			Node* node = new Node{};
			Vec3f boundMin, boundMax;
			json.getField("min", boundMin);
			json.getField("max", boundMax);
			node->bounding = Box3f(boundMin, boundMax);
			json.getField("splitAxis", node->splitAxis);
			if (auto child0 = json["child0"]) {
				node->children[0] = FromJson(child0);
			}
			if (auto child1 = json["child1"]) {
				node->children[1] = FromJson(child1);
			}
			return node;
		}
	};

	GPLevelCrossingKdTree(const std::string& filename);
	GPLevelCrossingKdTree(std::shared_ptr<GaussianProcess> gp,Box3f bounding,float stepSize,int maxDepth = -1,float zeroThreshold = 0.01);

	void Intersect(const Ray& ray,std::vector<Node*>& intersectedNodes) const;

	void Serialize(const std::string& filename);

	Node* Root() const { return root; }
private:
	std::tuple<double, double> calculateMinP(Box3f bounding);

	Node* buildRecursive(Box3f bounding, Node* parent,int depth);

private:
	Node* root = nullptr;
	std::shared_ptr<GaussianProcess> gp;
	float stepSize = 0;
	int maxDepth;
	float zeroThreshold = 0;

	float minGridSize = 0;

public:
	class MuDivSigma {
	public:
		MuDivSigma() = default;
		MuDivSigma(std::shared_ptr<GaussianProcess> gp) :gp(gp) {};
		template<typename Derived>
		double operator()(const Eigen::MatrixBase<Derived>& xval) const
		{
			assert(xval.size() == 3);
			Vec3d point = { xval(0),xval(1),xval(2) };
			auto derivativeNone = Derivative::None;
			auto derivativeFirst = Derivative::First;
			double cov = gp->cov_sym(&point, &derivativeNone, nullptr, Vec3d(), 1)(0, 0);
			double sigma = sqrt(cov);
			double mu = gp->mean(&point, &derivativeNone, nullptr, Vec3d(), 1)(0);

			return mu / sigma;
		}

		std::shared_ptr<GaussianProcess> gp;
	};
	class MinusMuDivSigma {
	public:
		MinusMuDivSigma() = default;
		MinusMuDivSigma(std::shared_ptr<GaussianProcess> gp) :gp(gp) {};
		template<typename Derived>
		double operator()(const Eigen::MatrixBase<Derived>& xval) const
		{
			assert(xval.size() == 3);
			Vec3d point = { xval(0),xval(1),xval(2) };
			auto derivativeNone = Derivative::None;
			auto derivativeFirst = Derivative::First;
			double cov = gp->cov_sym(&point, &derivativeNone, nullptr, Vec3d(), 1)(0, 0);
			double sigma = sqrt(cov);
			double mu = gp->mean(&point, &derivativeNone, nullptr, Vec3d(), 1)(0);

			return -mu / sigma;
		}

		std::shared_ptr<GaussianProcess> gp;
	};
	class Mu {
	public:
		Mu() = default;
		Mu(std::shared_ptr<GaussianProcess> gp) :gp(gp) {};
		template<typename Derived>
		double operator()(const Eigen::MatrixBase<Derived>& xval) const
		{
			assert(xval.size() == 3);
			Vec3d point = { xval(0),xval(1),xval(2) };
			auto derivativeNone = Derivative::None;
			double mu = gp->mean(&point, &derivativeNone, nullptr, Vec3d(), 1)(0);

			return mu;
		}

		std::shared_ptr<GaussianProcess> gp;
	};
	class MinusMu {
	public:
		MinusMu() = default;
		MinusMu(std::shared_ptr<GaussianProcess> gp) :gp(gp) {};
		template<typename Derived>
		double operator()(const Eigen::MatrixBase<Derived>& xval) const
		{
			assert(xval.size() == 3);
			Vec3d point = { xval(0),xval(1),xval(2) };
			auto derivativeNone = Derivative::None;
			double mu = gp->mean(&point, &derivativeNone, nullptr, Vec3d(), 1)(0);

			return -mu;
		}

		std::shared_ptr<GaussianProcess> gp;
	};
};
}
#endif