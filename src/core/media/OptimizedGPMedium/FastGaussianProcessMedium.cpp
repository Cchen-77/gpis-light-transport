#include "FastGaussianProcessMedium.hpp"

#include <stack>
namespace Tungsten {
void FastGaussianProcessMedium::fromJson(JsonPtr value, const Scene& scene) {
    GaussianProcessMedium::fromJson(value, scene);

    value.getField("step_size", stepSize);    
    value.getField("sample_points", samplePoints);
    value.getField("baked_kdtree", bakedKdTreeFileName);

    Path bakedFilePath = scene.GetSrcDir() + bakedKdTreeFileName;
    lcKdTree = std::make_shared<GPLevelCrossingKdTree>(bakedFilePath.asString());

    // we limit ourself to 
    _ctxt = GPCorrelationContext::Goldfish;
    _normalSamplingMethod = GPNormalSamplingMethod::ConditionedGaussian;

    if (gaussianProcess = std::dynamic_pointer_cast<GaussianProcess>(_gp);!gaussianProcess) {
        std::cout << "fast gaussian process meidum do not support CSG currently" << '\n';
        exit(-1);
    }
}

rapidjson::Value FastGaussianProcessMedium::toJson(Allocator& allocator) const
{
    return JsonObject{ GaussianProcessMedium::toJson(allocator), allocator,
            "type", "fast_gaussian_process",
            "step_size",stepSize,
            "baked_kdtree",JsonUtils::toJson(bakedKdTreeFileName,allocator)
    };
}

bool FastGaussianProcessMedium::sampleGradient(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& ip, MediumState& state, Vec3d& grad) const
{
    GPContextFunctionSpace& ctxt = *(GPContextFunctionSpace*)state.gpContext.get();
    auto rd = vec_conv<Vec3d>(ray.dir());

    ctxt.values->sampleGrad(state.lastGPId, ip, rd, ctxt.points.data(), ctxt.derivs.data(), sampler, grad);

    if (!std::isfinite(grad.avg())) {
        std::cout << "Sampled gradient invalid.\n";
        return false;
    }
    return true;
}

bool FastGaussianProcessMedium::sampleDistance(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, MediumSample& sample) const
{
    sample.emission = Vec3f(0.0f);
    auto r = ray;
    size_t matId = 0;

    double startT = r.nearT();
    if (!std::isfinite(r.farT())) {
        r.setFarT(startT + 2000);
    }

    float maxT = r.farT();

    if (state.bounce >= _maxBounce) {
        return false;
    }

    if (maxT == 0.f) {
        sample.t = maxT;
        sample.weight = Vec3f(1.f);
        sample.pdf = 1.0f;
        sample.exited = true;
        sample.p = ray.pos() + sample.t * ray.dir();
        sample.phase = _phaseFunction.get();
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
        double eps = 1e-6;
        double t = maxT;
        auto ro = vec_conv<Vec3d>(r.pos());
        auto rd = vec_conv<Vec3d>(r.dir()).normalized();

        bool intersected = false;
        auto root = lcKdTree->Root();
        std::stack<GPLevelCrossingKdTree::Node*> nodes;
        nodes.push(root);
        do {
            auto node = nodes.top();
            nodes.pop();
            Box3f& bounding = node->bounding;
            int splitAxis = node->splitAxis;
            float t0,t1;
            if (IntersectBox(bounding, r, &t0, &t1)) {
                if (node->isLeaf()) {
                    startT = t0;
                    do {
                        double maxRayDist = std::min((double)maxT, startT + stepSize * (samplePoints - 1)) - startT;
                        int determinedSamplePoints = std::ceil(maxRayDist / stepSize) + 1;

                        std::vector<Vec3d> points(determinedSamplePoints);
                        std::vector<Derivative> derivs(determinedSamplePoints);
                        std::vector<double> ts(determinedSamplePoints);

                        for (int i = 0; i < determinedSamplePoints; ++i) {
                            double rt = startT + stepSize * i;
                            if (i == 0)
                                rt = startT + stepSize * 0.1;
                            else if (i == determinedSamplePoints - 1)
                                rt = startT + maxRayDist;

                            ts[i] = rt;
                            points[i] = ro + rt * rd;
                            derivs[i] = Derivative::None;
                        }
                        
                        std::shared_ptr<GPRealNode> gpSamples;
                        if (state.firstScatter) {
                            gpSamples = _gp->sample(
                                points.data(), derivs.data(), determinedSamplePoints, nullptr, nullptr, 0, rd, 1, sampler);
                        }
                        else {
                            auto ctxt = std::static_pointer_cast<GPContextFunctionSpace>(state.gpContext);

                            if (ctxt->points.size() == 0) {
                                std::cerr << "Empty context!\n";
                            }
                            assert(ctxt->points.size() > 0);
                            Vec3d lastIntersectPt = ctxt->points[ctxt->points.size() - 1];
                            ctxt->values->applyMemory(_ctxt, rd);

                            std::array<Vec3d, 2> cond_pts = { lastIntersectPt, lastIntersectPt };
                            std::array<Derivative, 2> cond_deriv = { Derivative::None, Derivative::First };

                            gpSamples = _gp->sample_cond(
                                points.data(), derivs.data(), determinedSamplePoints, nullptr,
                                cond_pts.data(), ctxt->values.get(), cond_deriv.data(), cond_pts.size(), nullptr,
                                nullptr, 0,
                                rd, 1, sampler);
                        }

                        auto [sampleValues, sampleIds] = gpSamples->flatten();


                        double preV = sampleValues(0);
                        double preT = ts[0];
                        // to avoid error due to precision,we assume preV > 0 just like function space gp medium
                        preV = std::max(preV, 0.);

                        for (int i = 1; i < determinedSamplePoints; ++i) {
                            double curV = sampleValues(i);
                            double curT = ts[i];
                            if (curV < 0.) {
                                double offsetT = preV / (preV - curV);
                                t = lerp(preT, curT, offsetT);

                                derivs.resize(i + 2);
                                points.resize(i + 2);

                                gpSamples->makeIntersect(i, offsetT, preT - curT);

                                points[i] = ro + t * rd;
                                derivs[i] = Derivative::None;

                                points[i + 1] = ro + t * rd;
                                derivs[i + 1] = Derivative::First;

                                auto ctxt = std::make_shared<GPContextFunctionSpace>();
                                ctxt->derivs = std::move(derivs);
                                ctxt->points = std::move(points);
                                ctxt->values = gpSamples;
                                state.gpContext = ctxt;
                                state.lastGPId = sampleIds(i, 0);

                                intersected = true;
                                break;
                            }
                            preV = curV;
                            preT = curT;
                        }

                        if (intersected) {
                            Vec3d ip = ro + rd * t;
                            Vec3d grad;
                            if (!sampleGradient(sampler, r, ip, state, grad)) {
                                std::cout << "Failed to sample gradient.\n";
                                return false;
                            }

                            state.lastAniso = sample.aniso = grad;
                            state.firstScatter = false;

                            if (!std::isfinite(sample.aniso.avg())) {
                                sample.aniso = Vec3d(1.f, 0.f, 0.f);
                                std::cout << "Gradient invalid.\n";
                                return false;
                            }
                            sample.exited = false;
                            break;
                        }
                        else {
                            auto ctxt = std::make_shared<GPContextFunctionSpace>();
                            ctxt->derivs = std::move(derivs);
                            ctxt->points = std::move(points);
                            ctxt->values = gpSamples;
                            state.gpContext = ctxt;
                            state.lastGPId = sampleIds(sampleIds.size() - 1, 0);

                            startT = startT + maxRayDist;
                            if (maxT - startT < eps) {
                                break;
                            }
                            r.setNearT(startT);
                            //just like function space gp medium, we need sample graident if we want to continue.
                            Vec3d ip = ro + rd * startT;
                            Vec3d grad;
                            if (!sampleGradient(sampler, r, ip, state, grad)) {
                                std::cout << "Failed to sample gradient.\n";
                                return false;
                            }

                            state.lastAniso = sample.aniso = grad;
                            state.firstScatter = false;

                            if (!std::isfinite(sample.aniso.avg())) {
                                sample.aniso = Vec3d(1.f, 0.f, 0.f);
                                std::cout << "Gradient invalid.\n";
                                return false;
                            }
                        }
                    }while (!intersected && t1 - startT > eps);
                }
                else {
                    float splitCenter = (bounding.min()[splitAxis] + bounding.max()[splitAxis]) / 2.f;
                    GPLevelCrossingKdTree::Node* childFirst = node->children[0];
                    GPLevelCrossingKdTree::Node* childSecond = node->children[1];
                    if (ray.dir()[splitAxis] < 0.f) {
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
        } while (!intersected && maxT - startT > eps && !nodes.empty() );

        if (!intersected || maxT - startT < eps) {
            t = maxT;
            sample.exited = true;
        }

        if (!sample.exited) {
            if (sample.aniso.dot(vec_conv<Vec3d>(ray.dir())) > 0) {
                return false;
            }

            if (sample.aniso.lengthSq() < 0.0000001f) {
                sample.aniso = Vec3d(1.f, 0.f, 0.f);
                std::cout << "Gradient zero.\n";
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

    return true;
}



}
