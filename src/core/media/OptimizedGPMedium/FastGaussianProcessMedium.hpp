#ifndef FASTGAUSSIANPROCESSMEDIUM_HPP_
#define FASTGAUSSIANPROCESSMEDIUM_HPP_
#include "media/GaussianProcessMedium.hpp"
#include "GPLevelCrossingKdTree.hpp"

#include<atomic>
#include<mutex>
namespace Tungsten {
    class FastGaussianProcessMedium :public GaussianProcessMedium {
        double stepSize;
        int samplePoints;
        std::string bakedKdTreeFileName;
        std::shared_ptr<GPLevelCrossingKdTree> lcKdTree;

        std::shared_ptr<GaussianProcess> gaussianProcess;
    public:
        FastGaussianProcessMedium() :GaussianProcessMedium() {};
        FastGaussianProcessMedium(std::shared_ptr<GPSampleNode> gp,
            std::vector<std::shared_ptr<PhaseFunction>> phase,
            float materialSigmaA, float materialSigmaS, float density,
            GPCorrelationContext ctxt = GPCorrelationContext::Goldfish,
            GPIntersectMethod intersectMethod = GPIntersectMethod::GPDiscrete,
            GPNormalSamplingMethod normalSamplingMethod = GPNormalSamplingMethod::ConditionedGaussian,
            double stepSize = 0.05,int samplePoints = 8,
            std::shared_ptr<GPLevelCrossingKdTree> lcKdTree = nullptr) 
            :
            GaussianProcessMedium(gp, phase, materialSigmaA, materialSigmaS, density ,ctxt, intersectMethod, normalSamplingMethod),
            stepSize(stepSize),samplePoints(samplePoints),lcKdTree(lcKdTree)
        {}

        FastGaussianProcessMedium(std::shared_ptr<GPSampleNode> gp,
            float materialSigmaA, float materialSigmaS, float density,
            GPCorrelationContext ctxt = GPCorrelationContext::Goldfish,
            GPIntersectMethod intersectMethod = GPIntersectMethod::GPDiscrete,
            GPNormalSamplingMethod normalSamplingMethod = GPNormalSamplingMethod::ConditionedGaussian,
            double stepSize = 0.05,int samplePoints = 8,
            std::shared_ptr<GPLevelCrossingKdTree> lcKdTree = nullptr) 
            :
            GaussianProcessMedium(gp, { nullptr }, materialSigmaA, materialSigmaS, density, ctxt, intersectMethod, normalSamplingMethod),
            stepSize(stepSize), samplePoints(samplePoints), lcKdTree(lcKdTree)
        {}

        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;

        virtual bool sampleGradient(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& ip,
            MediumState& state,
            Vec3d& grad) const override;

        virtual bool sampleDistance(PathSampleGenerator& sampler, const Ray& ray,
            MediumState& state, MediumSample& sample) const override;


        /*virtual Vec3f transmittance(PathSampleGenerator& sampler, const Ray& ray, bool startOnSurface,
            bool endOnSurface, MediumSample * sample) const override;*/
    };

}
#endif