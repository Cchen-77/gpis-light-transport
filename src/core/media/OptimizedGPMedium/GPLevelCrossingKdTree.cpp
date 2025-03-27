#include "GPLevelCrossingKdTree.hpp"
#include <thread/ThreadUtils.hpp>
#include "thread/ThreadPool.hpp"
#include "thread/TaskGroup.hpp"

#include "pso-cpp/psocpp.h"
#include <boost/math/distributions/normal.hpp>
#include <stack>

#include "io/JsonDocument.hpp"
#include "fstream"
#include <rapidjson/stringbuffer.h>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
using Eigen::VectorXd;

namespace Tungsten {
GPLevelCrossingKdTree::GPLevelCrossingKdTree(const std::string& filename)
{
    const Path path(filename);
    InputStreamHandle in = FileUtils::openInputStream(path);

    JsonDocument document(path, FileUtils::streamRead<std::string>(in));

    root = Node::FromJson(document);
}
GPLevelCrossingKdTree::GPLevelCrossingKdTree(std::shared_ptr<GaussianProcess> gp, Box3f bounding, float stepSize,int maxDepth,float zeroThreshold):gp(gp),stepSize(stepSize),maxDepth(maxDepth),zeroThreshold(zeroThreshold)
{
    minGridSize = stepSize / std::sqrt(3);
    root = buildRecursive(bounding,nullptr,0);
}

static double fast_pow(double base, int exp) {
    double result = 1.0;
    bool neg_exp = exp < 0;
    exp = abs(exp);
    while (exp) {
        if (exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }
    return neg_exp ? 1.0 / result : result;
}

std::tuple<double,double> GPLevelCrossingKdTree::calculateMinP(Box3f bounding)
{

    pso::ParticleSwarmOptimization<double, MuDivSigma,
        pso::ExponentialDecrease2<double>> optimizer1;
    pso::ParticleSwarmOptimization<double, MinusMuDivSigma,
        pso::ExponentialDecrease2<double>> optimizer2;
    optimizer1.setObjective(MuDivSigma(gp));
    optimizer2.setObjective(MinusMuDivSigma(gp));

   /* pso::ParticleSwarmOptimization<double, Mu,
        pso::ExponentialDecrease2<double>> optimizer1;
    pso::ParticleSwarmOptimization<double, MinusMu,
        pso::ExponentialDecrease2<double>> optimizer2;*/

    optimizer1.setObjective(MuDivSigma(gp));
    optimizer2.setObjective(MinusMuDivSigma(gp));

    optimizer1.setMaxIterations(100);
    optimizer2.setMaxIterations(100);

    optimizer1.setMinParticleChange(1e-6);
    optimizer2.setMinParticleChange(1e-6);

    optimizer1.setMinFunctionChange(1e-6);
    optimizer2.setMinFunctionChange(1e-6);

    optimizer1.setThreads(1);
    optimizer2.setThreads(1);

    optimizer1.setVerbosity(0);
    optimizer2.setVerbosity(0);

    Eigen::MatrixXd bounds(2, 3);
    bounds << bounding.min()[0],bounding.min()[1],bounding.min()[2],
        bounding.max()[0],bounding.max()[1],bounding.max()[2];

    auto result1 = optimizer1.minimize(bounds, 100);
    auto result2 = optimizer2.minimize(bounds, 100);
    
    /*double minMu = result1.fval, minMinusMu = result2.fval;
    double cov = gp->_cov->operator()(Derivative::None, Derivative::None, Vec3d(0., 0., 0.), Vec3d(0., 0., 0.), Vec3d(), Vec3d());
    double sigma = sqrt(cov);
    double minMuDivSigma = minMu / sigma, minMinusMuDivSigma = minMinusMu / sigma;*/
    double minMuDivSigma = result1.fval, minMinusMuDivSigma = result2.fval;
    
    boost::math::normal_distribution<> normal(0, 1); 
    double pLessThanZero = boost::math::cdf(normal, minMuDivSigma);
    double pGreaterThanZero = 1. - boost::math::cdf(normal, -minMinusMuDivSigma);


    return std::make_tuple(pLessThanZero, pGreaterThanZero);
}

GPLevelCrossingKdTree::Node* Tungsten::GPLevelCrossingKdTree::buildRecursive(Box3f bounding,Node* parent,int depth)
{
    Vec3f extent = bounding.max() - bounding.min();
    float maxExtent = extent.max();

    auto node = new Node{};
    node->bounding = bounding;

    if (maxExtent < minGridSize) {
        return node;
    }
    if (depth == maxDepth) {
        return node;
    }

    int initSplitAxis = parent ? (parent->splitAxis + 1)%3 : 0;
    int determindedSplitAxis = initSplitAxis;
    Box3f determindedLBound, determindedRBound;
    float determindedCrossUBL = 1, determindedCrossUBR = 1;
    int maxEmptyChild = -1;

    for (int i = initSplitAxis; i < initSplitAxis + 3; ++i) {
        int splitAxis = i % 3;
        if (extent[splitAxis] < minGridSize) {
            continue;
        }
        float maxExtent = std::max(extent[splitAxis] / 2, std::max(extent[(splitAxis + 1) % 3], extent[(splitAxis + 2) % 3]));
        int steps = std::ceil(maxExtent / minGridSize) + 1;

        Vec3f lBoundingMax = bounding.max();
        Vec3f rBoundingMin = bounding.min();
        double axisCenter = (bounding.max()[splitAxis] + bounding.min()[splitAxis]) / 2;
        lBoundingMax[splitAxis] = axisCenter;
        rBoundingMin[splitAxis] = axisCenter;

        Box lBound = Box3f(bounding.min(), lBoundingMax);
        Box rBound = Box3f(rBoundingMin, bounding.max());

        auto [minLessL, minGreaterL] = calculateMinP(lBound);
        auto [minLessR, minGreaterR] = calculateMinP(rBound);

        double pCrossUpperBoundL = 1 - fast_pow(minLessL, steps) - fast_pow(minGreaterL, steps);
        double pCrossUpperBoundR = 1 - fast_pow(minLessR, steps) - fast_pow(minGreaterR, steps);

        int nEmptyChild = (pCrossUpperBoundL < zeroThreshold) + (pCrossUpperBoundR < zeroThreshold);

        if (nEmptyChild > maxEmptyChild) {
            node->splitAxis = splitAxis;
            maxEmptyChild = nEmptyChild;
            determindedLBound = lBound;
            determindedRBound = rBound;
            determindedCrossUBL = pCrossUpperBoundL;
            determindedCrossUBR = pCrossUpperBoundR;
        }
    }

    if (maxEmptyChild == 2) {
        delete node;
        return nullptr;
    }

    std::shared_ptr<TaskGroup> group;
    if (maxExtent > 10*stepSize) {
        if (determindedCrossUBL > zeroThreshold) {
            group = ThreadUtils::pool->enqueue([this,node,determindedLBound, determindedCrossUBL,depth](uint32, uint32, uint32) {
                node->children[0] = this->buildRecursive(determindedLBound, node,depth+1);
                }, 1, []() {});
        }
    }
    else {
        if (determindedCrossUBL > zeroThreshold) {
            node->children[0] = buildRecursive(determindedLBound, node, depth + 1);
        }
    }
    if (determindedCrossUBR > zeroThreshold) {
        node->children[1] = buildRecursive(determindedRBound, node, depth + 1);
    }

    if (group && !group->isDone()) {
        ThreadUtils::pool->yield(*group);
    }

    if (node->children[0] == nullptr && node->children[1] == nullptr) {
        delete node;
        return nullptr;
    }

    return node;
}

void GPLevelCrossingKdTree::Intersect(const Ray& ray,std::vector<Node*>& intersectdNodes) const
{
    if (!root) {
        return;
    }
    std::stack<Node*> nodes;
    nodes.push(root);
    do {
        Node* node = nodes.top();
        nodes.pop();
        Box3f& bounding = node->bounding;
        int splitAxis = node->splitAxis;
        if (IntersectBox(bounding, ray, nullptr, nullptr)) {
            if (node->isLeaf()) {
                intersectdNodes.push_back(node);
            }
            else{
                float splitCenter = (bounding.min()[splitAxis] + bounding.max()[splitAxis]) / 2.f;
                Node* childFirst = node->children[0];
                Node* childSecond = node->children[1];
                if (splitCenter - ray.pos()[splitAxis] * ray.dir()[splitAxis] < 0.f) {
                    std::swap(childFirst, childSecond);
                }
                if (childSecond) {
                    nodes.push(childSecond);
                }
                if (childFirst) {
                    nodes.push(childFirst);
                }
            }
        }

    } while (!nodes.empty());
}
void GPLevelCrossingKdTree::Serialize(const std::string& filename)
{
    Path path(filename);
    OutputStreamHandle out = FileUtils::openOutputStream(path);
    if (!out) {
        std::cout << "LCKdTree: failed to open output stream" << '\n';
    }

    rapidjson::Document document;
    document.SetObject();
    *(static_cast<rapidjson::Value*>(&document)) = root->toJson(document.GetAllocator());

    FileUtils::streamWrite(out, JsonUtils::jsonToString(document));
}

}
