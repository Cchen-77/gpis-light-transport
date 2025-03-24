#ifndef NRAGAUSSIANPROCESSMEDIUM_HPP_
#define NRAGAUSSIANPROCESSMEDIUM_HPP_
#include"FunctionSpaceGaussianProcessMedium.hpp"

#include<atomic>
#include<mutex>
#define ENABLE_PROFILE 1
namespace Tungsten {
class NRAGaussianProcessMeidum :public FunctionSpaceGaussianProcessMedium {
    double NRAConditionFineCheckingDistance;
    int NRAConditionFineCheckingSamplePoints;

    GaussianProcess* gaussianProcess = nullptr;
    bool degenerated = false;
    bool enableNRA = true;
    // use fine checking to help skip empty space even when nra is not enable
    bool finerSkip = true;
    bool totallyOrigin = false;
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
#if(ENABLE_PROFILE)
    ~NRAGaussianProcessMeidum() {
        std::cout << "=================[info]=================" << '\n';
        std::cout << "total Sample Distance: " << sampleDistanceCount.load() << '\n';
        std::cout << "skip: " << skipCount.load() << '\n';
        std::cout << "fc skip Count: " << fcSkipCount.load() << '\n';
        std::cout << "skip FC iter: " << skipFCCount.load() << '\n';
        std::cout << "nra optimized:" << nraOptimizedCount.load() << '\n';
        std::cout << "nra FC iter:" << nraFCCount.load() << '\n';
        std::cout << "origin intersect:" << originCount.load() << '\n';
        std::cout << "NRA condtion 1 failed: " << fail1Count.load() << '\n';
        std::cout << "NRA condtion 2 failed: " << fail2Count.load() << '\n';
        
        std::cout << '\n';
        std::cout << "total Sample Distance Time: " << totalTime.load() << "s\n";
        std::cout << "overhead time: " << overheadTime.load() << "s\n";
        std::cout << "fine checking time:" << fineCheckingTime.load() << "s\n";
        std::cout << "skip time:" << skipTime.load() << "s\n";
        std::cout << "skip checking time:" << skipCheckingTime.load() << "s\n";
        std::cout << "fine checking skip time: " << fineCheckingSkipTime.load() << "s\n";
        std::cout << "fine checking skip sdf time: " << fineCheckingSkipSDFTime.load() << "s\n";
        std::cout << "nra time:" << nraCheckingTime.load() +nraSampleGraidentTime.load() <<"s " << "| avg: " << (nraCheckingTime.load() + nraSampleGraidentTime.load()) / nraOptimizedCount.load() << "s\n";
        std::cout << "cdf time:" << cdfTime.load() << "s\n";
        std::cout << "nra sample gradient time: " << nraSampleGraidentTime.load() << "s " << "| avg: " << nraSampleGraidentTime.load() / nraOptimizedCount.load() << "s\n";
        std::cout << "origin time:" << originTime.load() << "s " << "| avg: " << originTime.load() / originCount.load() << "s\n";
        std::cout << "origin sample gradient time: " << originSampleGraidentTime.load() << "s " <<"| avg: "<< originSampleGraidentTime.load()/originCount.load()<<"s\n";
        std::cout << "origin time with overhead:" << originTimeWithOverhead.load() << "s\n";
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

#if(ENABLE_PROFILE)
    mutable std::atomic<int> sampleDistanceCount = 0;
    mutable std::atomic<int> skipCount = 0;
    mutable std::atomic<int> fcSkipCount = 0.;
    mutable std::atomic<int> skipFCCount = 0;
    mutable std::atomic<int> nraOptimizedCount = 0;
    mutable std::atomic<int> nraFCCount = 0;
    mutable std::atomic<int> originCount = 0;
    mutable std::atomic<int> fail1Count = 0;
    mutable std::atomic<int> fail2Count = 0;

    mutable std::atomic<double> overheadTime = 0.;
    mutable std::atomic<double> fineCheckingTime = 0.;
    mutable std::atomic<double> skipTime = 0.;
    mutable std::atomic<double> fineCheckingSkipTime = 0.;
    mutable std::atomic<double> fineCheckingSkipSDFTime = 0.;
    mutable std::atomic<double> skipCheckingTime = 0.;
    mutable std::atomic<double> nraCheckingTime = 0.;
    mutable std::atomic<double> cdfTime = 0.;
    mutable std::atomic<double> nraSampleGraidentTime = 0.;
    mutable std::atomic<double> originTime = 0.;
    mutable std::atomic<double> originTimeWithOverhead = 0.;
    mutable std::atomic<double> originSampleGraidentTime = 0.;
    mutable std::atomic<double> totalTime = 0.;

    mutable std::mutex logMutex;
#endif
};

}
#endif