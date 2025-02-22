#ifndef NRAGAUSSIANPROCESSMEDIUM_HPP_
#define NRAGAUSSIANPROCESSMEDIUM_HPP_
#include"FunctionSpaceGaussianProcessMedium.hpp"

#include<atomic>
#define ENABLE_COUNTER 1
#define ENABLE_TIMER 1
namespace Tungsten {
class NRAGaussianProcessMeidum :public FunctionSpaceGaussianProcessMedium {
    double NRAConditionFineCheckingDistance;
    int NRAConditionFineCheckingSamplePoints;

    GaussianProcess* gaussianProcess = nullptr;
    bool degenerated = false;
    bool enableNRA = true;
public:
    NRAGaussianProcessMeidum() :FunctionSpaceGaussianProcessMedium(), NRAConditionFineCheckingDistance(24), NRAConditionFineCheckingSamplePoints(32) {};
    NRAGaussianProcessMeidum(std::shared_ptr<GPSampleNode> gp,
        std::vector<std::shared_ptr<PhaseFunction>> phase,
        float materialSigmaA, float materialSigmaS, float density, int samplePoints,
        GPCorrelationContext ctxt = GPCorrelationContext::Goldfish,
        GPIntersectMethod intersectMethod = GPIntersectMethod::GPDiscrete,
        GPNormalSamplingMethod normalSamplingMethod = GPNormalSamplingMethod::ConditionedGaussian,
        double stepSizeCov = 0, double stepSize = 0, double skipSpace = 0,
        int NRAConditionFineCheckingDistance = 24,int NRAConditionFineCheckingSamplePoints = 32) :
        FunctionSpaceGaussianProcessMedium(gp, phase, materialSigmaA, materialSigmaS, density, samplePoints, ctxt, intersectMethod, normalSamplingMethod, stepSizeCov, stepSize, skipSpace)
        ,NRAConditionFineCheckingDistance(NRAConditionFineCheckingDistance), NRAConditionFineCheckingSamplePoints(NRAConditionFineCheckingSamplePoints)
    {
        
    }
#if(ENABLE_COUNTER||ENABLE_TIMER)
    ~NRAGaussianProcessMeidum() {
        std::cout << "=================[info]=================" << '\n';
#if(ENABLE_COUNTER)
        std::cout << "total Sample Distance: " << sampleDistanceCount.load() << '\n';
        std::cout << "skip: " << skipCount.load() << '\n';
        std::cout << "nra optimized:" << nraOptimizedCount.load() << '\n';
        std::cout << "origin intersect:" << originCount.load() << '\n';
        std::cout << "fine checking times:" << fcCount.load() << '\n';
        std::cout << "function space intersect GP times:" << gpsampleCount.load() << '\n';
        std::cout << '\n';
#endif
#if(ENABLE_TIMER)
        std::cout << "total Sample Distance Time: " << totalTime.load() << "s\n";
        std::cout << "overhead time:" << overheadTime.load() << "s\n";
        std::cout << "cdf caculate time: " << cdfTime.load() << "s\n";
        std::cout << "nra optimized time:" << nraOptimizedTime.load() << "s\n";
        std::cout << "origin time:" << originTime.load() << "s\n";
#endif
        std::cout << "=======================================" << '\n';
    }
#endif

    NRAGaussianProcessMeidum(std::shared_ptr<GPSampleNode> gp,
        float materialSigmaA, float materialSigmaS, float density, int samplePoints,
        GPCorrelationContext ctxt = GPCorrelationContext::Goldfish,
        GPIntersectMethod intersectMethod = GPIntersectMethod::GPDiscrete,
        GPNormalSamplingMethod normalSamplingMethod = GPNormalSamplingMethod::ConditionedGaussian,
        double stepSizeCov = 0, double stepSize = 0, double skipSpace = 0,
        int NRAConditionFineCheckingDistance = 8, int NRAConditionFineCheckingSamplePoints = 32) :
        FunctionSpaceGaussianProcessMedium(gp, {nullptr}, materialSigmaA, materialSigmaS, density, samplePoints, ctxt, intersectMethod, normalSamplingMethod, stepSizeCov, stepSize, skipSpace)
        , NRAConditionFineCheckingDistance(NRAConditionFineCheckingDistance), NRAConditionFineCheckingSamplePoints(NRAConditionFineCheckingSamplePoints)
    {}

    virtual void fromJson(JsonPtr value, const Scene& scene) override;
    virtual rapidjson::Value toJson(Allocator& allocator) const override;

    virtual bool sampleDistance(PathSampleGenerator& sampler, const Ray& ray,
        MediumState& state, MediumSample& sample) const override;
    virtual Vec3f transmittance(PathSampleGenerator& sampler, const Ray& ray, bool startOnSurface, bool endOnSurface, MediumSample* sample) const override;


    /*virtual Vec3f transmittance(PathSampleGenerator& sampler, const Ray& ray, bool startOnSurface,
        bool endOnSurface, MediumSample * sample) const override;*/

#if(ENABLE_COUNTER)
    mutable std::atomic<int> sampleDistanceCount = 0;
    mutable std::atomic<int> skipCount = 0;
    mutable std::atomic<int> nraOptimizedCount = 0;
    mutable std::atomic<int> originCount = 0;

    mutable std::atomic<int> fcCount = 0;
    mutable std::atomic<int> gpsampleCount = 0;
#endif
#if(ENABLE_TIMER)
    mutable std::atomic<double> overheadTime = 0.;
    mutable std::atomic<double> cdfTime = 0.;
    mutable std::atomic<double> originTime = 0.;
    mutable std::atomic<double> nraOptimizedTime = 0.;
    mutable std::atomic<double> totalTime = 0.;
#endif
};

}
#endif