#include "NRAGaussianProcessMeidum.hpp"
#include "Timer.hpp"
#include <boost/math/special_functions/erf.hpp>
namespace Tungsten {
void NRAGaussianProcessMeidum::fromJson(JsonPtr value, const Scene& scene) {
	FunctionSpaceGaussianProcessMedium::fromJson(value, scene);
    value.getField("nra_finechecking_distance", NRAConditionFineCheckingDistance);
    value.getField("nra_findechecking_sample_points", NRAConditionFineCheckingSamplePoints);
    value.getField("enable_nra", enableNRA);
    value.getField("finer_skip", finerSkip);
    value.getField("totally_origin", totallyOrigin);

    gaussianProcess = dynamic_cast<GaussianProcess*>(_gp.get());
    if (!gaussianProcess) {
        degenerated = true;
        std::cout << "NRAGaussianProcess degnerate to FunctionSpaceGaussianProcess since provided stochastic process is not a Gaussian process"<<'\n';
    }
    if (_ctxt != GPCorrelationContext::Goldfish) {
        degenerated = true;
        std::cout << "NRAGaussianProcess degnerate to FunctionSpaceGaussianProcess since memory model is not Renewal or Renewal+" << '\n';
    }
    if (gaussianProcess&&!dynamic_cast<StationaryCovariance*>(gaussianProcess->_cov.get())) {
        degenerated = true;
        std::cout << "NRAGaussianProcess degnerate to FunctionSpaceGaussianProcess since gaussian process has a non-stationary kernel" << '\n';
    }
}

rapidjson::Value NRAGaussianProcessMeidum::toJson(Allocator& allocator) const
{
    return JsonObject{ GaussianProcessMedium::toJson(allocator), allocator,
            "type", "nra_gaussian_process",
            "nra_finechecking_distance", NRAConditionFineCheckingDistance,
            "nra_findechecking_sample_points", NRAConditionFineCheckingSamplePoints,
    };
}
#if(ENABLE_PROFILE)
void atomic_add(std::atomic<double>& a, double b) {
    for (double g = a; !a.compare_exchange_weak(g, g + b););
}
#endif

bool NRAGaussianProcessMeidum::sampleDistance(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, MediumSample& sample) const
{
    
    if (degenerated) {
        return FunctionSpaceGaussianProcessMedium::sampleDistance(sampler, ray, state, sample);
    }

#if (ENABLE_PROFILE)
    Timer totalTimer;
    totalTimer.start();
    sampleDistanceCount.fetch_add(1);
#endif 

    sample.emission = Vec3f(0.0f);
    auto r = ray;
    size_t matId = 0;

    double startT = r.nearT();
    if (!std::isfinite(r.farT())) {
        r.setFarT(startT + 2000);
    }

    float maxT = r.farT();

    if (state.bounce >= _maxBounce) {

#if (ENABLE_PROFILE)
        totalTimer.stop();
        atomic_add(totalTime, totalTimer.elapsed());
#endif
        return false;
    }

    if (maxT == 0.f) {
        sample.t = maxT;
        sample.weight = Vec3f(1.f);
        sample.pdf = 1.0f;
        sample.exited = true;
        sample.p = ray.pos() + sample.t * ray.dir();
        sample.phase = _phaseFunction.get();

#if (ENABLE_PROFILE)
        totalTimer.stop();
        atomic_add(totalTime,totalTimer.elapsed());
#endif
        return true;
    }

    if (_absorptionOnly) {
        if (maxT == Ray::infinity())
            return false;
        sample.t = maxT;
        sample.weight = transmittance(sampler, ray, state.firstScatter, true, &sample);
        sample.pdf = 1.0f;
        sample.exited = true;
    }
    else {
#if (ENABLE_PROFILE)
        Timer overheadTimer;
        overheadTimer.start();
#endif
        double t = maxT;

        auto ro = vec_conv<Vec3d>(r.pos());
        auto rd = vec_conv<Vec3d>(r.dir()).normalized();

        // 3-sigma SDF marching to approch 'surface'
        GaussianProcess conditionedGP = *gaussianProcess;
        if (!state.firstScatter&&(enableNRA|| finerSkip)&&!totallyOrigin) {
            auto ctxt = std::static_pointer_cast<GPContextFunctionSpace>(state.gpContext);
            Vec3d lastIntersectPt = ctxt->points[ctxt->points.size() - 1];
            auto lastValues = std::static_pointer_cast<GPRealNodeValues>(ctxt->values);
            switch (_ctxt) {
            case GPCorrelationContext::Dori: {
                if (lastValues->_isIntersect)
                    conditionedGP._globalCondValues.push_back(lastValues->_values(lastValues->_values.size() - 2, 0));
                else
                    conditionedGP._globalCondValues.push_back(lastValues->_values(lastValues->_values.size() - 1, 0));
                conditionedGP._globalCondPs.push_back(lastIntersectPt);
                conditionedGP._globalCondDerivs.push_back(Derivative::None);
                conditionedGP._globalCondDerivDirs.push_back({});
                break;
            }
            case GPCorrelationContext::Goldfish: {
                if (lastValues->_isIntersect)
                    conditionedGP._globalCondValues.push_back(lastValues->_values(lastValues->_values.size() - 2, 0));
                else
                    conditionedGP._globalCondValues.push_back(lastValues->_values(lastValues->_values.size() - 1, 0));
                conditionedGP._globalCondPs.push_back(lastIntersectPt);
                conditionedGP._globalCondDerivs.push_back(Derivative::None);
                conditionedGP._globalCondDerivDirs.push_back({});

                conditionedGP._globalCondValues.push_back(lastValues->_sampledGrad.dot(rd));
                conditionedGP._globalCondPs.push_back(lastIntersectPt);
                conditionedGP._globalCondDerivs.push_back(Derivative::First);
                conditionedGP._globalCondDerivDirs.push_back(rd);
                break;
            }
            }
            conditionedGP.setConditioning(conditionedGP._globalCondPs, conditionedGP._globalCondDerivs, conditionedGP._globalCondDerivDirs, conditionedGP._globalCondValues);
        }
        double sigma = std::sqrt(gaussianProcess->_cov->operator()(Derivative::None, Derivative::None, Vec3d{0.,0.,0.}, Vec3d{0.,0.,0.}, {}, {}));
        double sigma3 = 3 * sigma;
        
        bool useNoReturnApproximation = false;
        double fcDistance = NRAConditionFineCheckingDistance;
        Derivative derivNone = Derivative::None;
        Derivative derivFirst = Derivative::First;
        
#if(ENABLE_PROFILE)
        double skipCheckingTime = 0.;
        int fcCount = 0.;
#endif 

        while (startT<maxT&&!totallyOrigin) {
            Vec3d pos = ro + rd * startT;
            double distance = std::abs(gaussianProcess->_mean->operator()(Derivative::None, pos, {}));
            if (distance <= sigma3) {
                if (enableNRA|| finerSkip) {
#if (ENABLE_PROFILE)
                    Timer nraCheckingTimer;
                    nraCheckingTimer.start();
#endif
                    useNoReturnApproximation = enableNRA;
                    bool noCollision = true;
                    double fcStepSize = std::min(fcDistance, maxT - startT) / (NRAConditionFineCheckingSamplePoints);
                    double lastP = 0.;
                    double lastT = startT;
                    double u = sampler.next1D();
                    double invCDFu = 0.;
                    std::vector<Vec3d> positions(NRAConditionFineCheckingSamplePoints);
                    std::vector<Derivative> derivatives(NRAConditionFineCheckingSamplePoints, Derivative::None);
                    for (int i = 1; i <= NRAConditionFineCheckingSamplePoints; ++i) {
                        double curT = startT + i * fcStepSize;
                        positions[i - 1] = ro + rd * curT;
                    }

                    auto covs = gaussianProcess->cov_sym(positions.data(), derivatives.data(), nullptr, Vec3d(), NRAConditionFineCheckingSamplePoints);
                    auto means = gaussianProcess->mean(positions.data(), derivatives.data(), nullptr, Vec3d(), NRAConditionFineCheckingSamplePoints);
                    for (int i = 1; i <= NRAConditionFineCheckingSamplePoints; ++i) {
#if (ENABLE_PROFILE)
                        ++fcCount;
#endif
                        double curT = startT + i * fcStepSize;
                        Vec3d curPos = ro + rd * curT;
                        /*double P = 0.;
                        double stddev = (conditionedGP.cov_sym(&curPos, &derivNone, nullptr, Vec3d(), 1)(0, 0));
                        double mu = conditionedGP.mean(&curPos, &derivNone, nullptr, Vec3d(), 1)(0);
                        if (stddev < 0 || std::abs(stddev) < 1e-10) {
                            P = mu < 0;
                        }
                        else {
                            stddev = sqrt(stddev);
                            P = 0.5 * (1 + boost::math::erf((0 - mu) / (stddev * sqrt(2))));
                        }*/
                        double stddev = std::sqrt(covs(i - 1, i - 1));
                        double mu = means(i - 1);
                        double P = 0.5 * (1 + boost::math::erf((0 - mu) / (stddev * sqrt(2))));

                        if (P < lastP) {
                            if(lastP - P < 0.01){
                                P = lastP;
                            }
                            else {
                                useNoReturnApproximation = false;
                                /*std::lock_guard lock(logMutex);
                                std::cout << P << '<' << lastP << '\n';*/
                            }
                        }
                        if (useNoReturnApproximation && u<P && u>lastP) {
                            invCDFu = lerp(lastT, curT, (u - lastP) / (P - lastP));
                        }
                        if (P > 0.01) {
                            noCollision = false;
                        }
                        lastP = P;
                        lastT = curT; 
                        if (!noCollision && !useNoReturnApproximation) {
                            break;
                        }
                        if (lastP > 0.99) {
                            break;
                        }
                    }
#if (ENABLE_PROFILE)
                    nraCheckingTimer.stop();
                    atomic_add(fineCheckingTime, nraCheckingTimer.elapsed());
#endif
                    if (noCollision) {
                        startT += fcDistance + 1e-4;
                        skipCheckingTime += nraCheckingTimer.elapsed();
                        continue;
                    }
                    if (useNoReturnApproximation) {
                        if (lastP < 0.99) {
                            useNoReturnApproximation = false;
#if (ENABLE_PROFILE)
                            fail2Count.fetch_add(1);
#endif
                           /* std::lock_guard lock(logMutex);
                            std::cout << lastP << '\n';*/
                        }
                        else {
#if (ENABLE_PROFILE)
                            atomic_add(nraCheckingTime, nraCheckingTimer.elapsed());
#endif
                            t = u > lastP ? lastT : invCDFu;
                            break;
                        }
                    }
#if (ENABLE_PROFILE)
                    else {
                        fail1Count.fetch_add(1);
                    }
#endif
                }
                break;
            }
            startT += std::max(1e-6, distance - sigma3);
        }
#if (ENABLE_PROFILE)
        overheadTimer.stop();
        atomic_add(overheadTime, overheadTimer.elapsed());
#endif
        if(startT>maxT){
#if(ENABLE_PROFILE)
            skipCount.fetch_add(1);
            atomic_add(skipTime, overheadTimer.elapsed());
            atomic_add(this->skipCheckingTime, skipCheckingTime);
            if (skipCheckingTime > 0) {
                atomic_add(fineCheckingSkipTime, overheadTimer.elapsed());
                atomic_add(fineCheckingSkipSDFTime, overheadTimer.elapsed() - skipCheckingTime);
                fcSkipCount.fetch_add(1);
                skipFCCount.fetch_add(fcCount);
            }
#endif
            sample.exited = true;
            t = maxT;
        }
        else if (useNoReturnApproximation) {
            t = min(float(t), maxT);
            Vec3d ip = ro + t * rd;
            Eigen::MatrixXd v(2, 1);
            // calculate mean(dot X | X = 0)
            double kCC = gaussianProcess->cov_sym(&ip, &derivNone, nullptr, {}, 1)(0, 0);
            double kxC = gaussianProcess->cov(&ip, &ip, &derivFirst, &derivNone, nullptr, nullptr, rd, 1, 1)(0, 0);
            double kCx = gaussianProcess->cov(&ip, &ip, &derivNone, &derivFirst, nullptr, nullptr, rd, 1, 1)(0, 0);
            v(0) = 0.;
            v(1) = gaussianProcess->mean(&ip, &derivFirst, nullptr, rd, 1)(0) - kxC * kCx / kCC * gaussianProcess->mean(&ip, &derivNone, nullptr, rd, 1)(0);

            auto newValues = std::make_shared<GPRealNodeValues>(v,gaussianProcess);
            auto newCtxt = std::make_shared<GPContextFunctionSpace>();
            newCtxt->points = { ip,ip };
            newCtxt->derivs = { Derivative::None,Derivative::First };
            newCtxt->values = newValues;

            state.gpContext = newCtxt;
            state.lastGPId = gaussianProcess->_id;

            newValues->_isIntersect = true;
            newValues->_gp = gaussianProcess;
            Vec3d grad;
#if(ENABLE_PROFILE)
            nraFCCount.fetch_add(fcCount);
            nraOptimizedCount.fetch_add(1);
            Timer nraSampleGraidentTimer;
            nraSampleGraidentTimer.start();
#endif
            newValues->sampleGrad(gaussianProcess->_id, ip, rd, newCtxt->points.data(), newCtxt->derivs.data(), sampler, grad);
#if (ENABLE_PROFILE)
            nraSampleGraidentTimer.stop();
            atomic_add(nraSampleGraidentTime, nraSampleGraidentTimer.elapsed());
#endif

            state.lastAniso = sample.aniso = grad;
            sample.exited = false;
            state.firstScatter = false;

        }
        else {
#if(ENABLE_PROFILE)
            originCount.fetch_add(1);
            Timer originTimer;
            originTimer.start();
#endif
            // Handle the "ray marching" case
            // I.e. we want to allow the intersect function to not handle the whole ray
            // In that case it will tell us it didn't intersect, but t will be less than ray.farT()
            do {
                r.setNearT((float)startT);
                sample.exited = !intersect(sampler, r, state, t);

                // We sample a gradient if:
                // (1) The ray did intersect a surface
                // (2) The ray did not intersect a surface, but we'll need to continue
                if (t < maxT) {

                    Vec3d ip = ro + rd * t;

                    Vec3d grad;
#if(ENABLE_PROFILE)
                    Timer originSampleGraidentTimer;
                    originSampleGraidentTimer.start();
#endif
                    if (!sampleGradient(sampler, ray, ip, state, grad)) {
                        std::cout << "Failed to sample gradient.\n";
                        return false;
                    }
#if(ENABLE_PROFILE)
                    originSampleGraidentTimer.stop();
                    atomic_add(originSampleGraidentTime, originSampleGraidentTimer.elapsed());
#endif

                    state.lastAniso = sample.aniso = grad;
                    state.firstScatter = false;

                    if (!std::isfinite(sample.aniso.avg())) {
                        sample.aniso = Vec3d(1.f, 0.f, 0.f);
                        std::cout << "Gradient invalid.\n";
                        return false;
                    }
                }

                startT = t;

                // We only keep going in the case where we haven't finished processing the ray yet.
            } while (t < maxT && sample.exited);
#if(ENABLE_PROFILE)
            originTimer.stop();
            atomic_add(originTime, originTimer.elapsed());
            atomic_add(originTimeWithOverhead, originTimer.elapsed() + overheadTimer.elapsed());
#endif

        }

        if (!sample.exited) {
            if (sample.aniso.dot(vec_conv<Vec3d>(ray.dir())) > 0) {
                //std::cout << "Sampled gradient at intersection point points in the wrong direction. "<< sample.aniso.dot(vec_conv<Vec3d>(ray.dir())) << "\n";
#if (ENABLE_PROFILE)
                totalTimer.stop();
                atomic_add(totalTime, totalTimer.elapsed());
#endif
                return false;
            }

            if (sample.aniso.lengthSq() < 0.0000001f) {
                sample.aniso = Vec3d(1.f, 0.f, 0.f);
                std::cout << "Gradient zero.\n";

#if (ENABLE_PROFILE)
                totalTimer.stop();
                atomic_add(totalTime, totalTimer.elapsed());
#endif
                return false;
            }

            sample.weight = sample.continuedWeight = vec_conv<Vec3f>(_gp->color(ro + rd * t));
            sample.emission = vec_conv<Vec3f>(_gp->emission(ro + rd * t));
        }
        else {
            sample.weight = sample.continuedWeight = Vec3f(1.f);
        }

        sample.t = min(float(t), maxT);
        sample.continuedT = float(t);
        sample.weight *= sigmaS(ray.pos() + sample.t * ray.dir()) / sigmaT(ray.pos() + sample.t * ray.dir());
        sample.continuedWeight *= sigmaS(ray.pos() + sample.continuedT * ray.dir()) / sigmaT(ray.pos() + sample.continuedT * ray.dir());
        sample.pdf = 1;

        state.lastAniso = sample.aniso;
        state.advance();
    }
    sample.p = ray.pos() + sample.t * ray.dir();

    sample.phase = _phaseFunctions[state.lastGPId].get();
    sample.gpId = state.lastGPId;
    sample.ctxt = state.gpContext.get();

#if (ENABLE_PROFILE)
    totalTimer.stop();
    atomic_add(totalTime, totalTimer.elapsed());
#endif

    return true;
}

Vec3f NRAGaussianProcessMeidum::transmittance(PathSampleGenerator& sampler, const Ray& ray, bool startOnSurface, bool endOnSurface, MediumSample* sample) const
{
    return FunctionSpaceGaussianProcessMedium::transmittance(sampler, ray, startOnSurface, endOnSurface, sample);
}



}
