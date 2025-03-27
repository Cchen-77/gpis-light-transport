#include <core/math/GaussianProcess.hpp>
#include <core/media/GaussianProcessMedium.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/io/StringUtils.hpp>
#include <core/math/Ray.hpp>
#include <thread/ThreadUtils.hpp>
#include <fstream>
#include <cfloat>
#include <io/Scene.hpp>


#include "Timer.hpp"

#ifdef OPENVDB_AVAILABLE
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#endif

#include <core/media/OptimizedGPMedium/GPLevelCrossingKdTree.hpp>


using namespace Tungsten;

int main(int argc, char** argv) {

	ThreadUtils::startThreads(16);

	EmbreeUtil::initDevice();

#ifdef OPENVDB_AVAILABLE
	openvdb::initialize();
#endif

	auto scenePath = Path(argv[1]);
	auto outputPath = Path(argv[2]);
	std::cout << "loading scene: "<<scenePath << "...\n";

	Scene* scene = nullptr;
	TraceableScene* tscene = nullptr;
	try {
		scene = Scene::load(scenePath);
		scene->loadResources();
		tscene = scene->makeTraceable();
	}
	catch (std::exception& e) {
		std::cout << e.what();
		return -1;
	}

	std::shared_ptr<GaussianProcessMedium> gp_medium = std::static_pointer_cast<GaussianProcessMedium>(scene->media()[0]);

	auto gp = std::dynamic_pointer_cast<GaussianProcess>(gp_medium->_gp);

	auto processBox = scene->findPrimitive("processBox");

	Box3f bounding = processBox->bounds();
	bounding.max() -= Vec3f(0.01f, 0.01f, 0.01f);
	bounding.min() += Vec3f(0.01f, 0.01f, 0.01f);

	std::cout << "start build lckdtree with bounding: " << bounding << "...\n";

	Timer timer;
	timer.start();

	auto kdtree = std::make_shared<GPLevelCrossingKdTree>(gp, bounding,0.05);

	timer.stop();

	std::cout << "build finish. build time: " << StringUtils::durationToString(timer.elapsed())<<'\n';

	std::cout << "start serialize lckdtree...";
	timer.start();

	kdtree->Serialize(outputPath.asString());

	timer.stop();
	std::cout << "serialize finish. serialize time: " << StringUtils::durationToString(timer.elapsed()) << '\n';

	std::cout << "start test loading lckdtree...\n";
	timer.start();

	auto newKdTree = std::make_shared<GPLevelCrossingKdTree>(outputPath.asString());

	timer.stop();
	std::cout << "load finish. loading time: " << StringUtils::durationToString(timer.elapsed()) << '\n';

}
