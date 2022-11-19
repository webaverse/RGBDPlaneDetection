#include "plane_detection.h"

PlaneDetection *plane_detection_ptr = nullptr;

//-----------------------------------------------------------------
// MRF energy functions
MRF::CostVal dCost(int pix, int label)
{
	return plane_detection_ptr->dCost(pix, label);
}

MRF::CostVal fnCost(int pix1, int pix2, int i, int j)
{
	return plane_detection_ptr->fnCost(pix1, pix2, i, j);
}

void runMRFOptimization(PlaneDetection &plane_detection)
{
	plane_detection_ptr = &plane_detection;

	DataCost *data = new DataCost(dCost);
	SmoothnessCost *smooth = new SmoothnessCost(fnCost);
	EnergyFunction *energy = new EnergyFunction(data, smooth);
	int width = plane_detection.cloud.w, height = plane_detection.cloud.h;
	MRF* mrf = new Expansion(width * height, plane_detection.plane_num_ + 1, energy);
	// Set neighbors for the graph
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			int pix = row * width + col;
			if (col < width - 1) // horizontal neighbor
				mrf->setNeighbors(pix, pix + 1, 1);
			if (row < height - 1) // vertical
				mrf->setNeighbors(pix, pix + width, 1);
			if (row < height - 1 && col < width - 1) // diagonal
				mrf->setNeighbors(pix, pix + width + 1, 1);
		}
	}
	mrf->initialize();
	mrf->clearAnswer();
	float t;
	mrf->optimize(5, t);  // run for 5 iterations, store time t it took 
	MRF::EnergyVal E_smooth = mrf->smoothnessEnergy();
	MRF::EnergyVal E_data = mrf->dataEnergy();
	cout << "Optimized Energy: smooth = " << E_smooth << ", data = " << E_data << endl;
	cout << "Time consumed in MRF: " << t << endl;

	// Get MRF result
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			int pix = row * width + col;
			plane_detection.opt_seg_img_.at<cv::Vec3b>(row, col) = plane_detection.plane_colors_[mrf->getLabel(pix)];
			plane_detection.opt_membership_img_.at<int>(row, col) = mrf->getLabel(pix);
		}
	}
	delete mrf;
	delete energy;
	delete smooth;
	delete data;
}
//-----------------------------------------------------------------


void printUsage()
{
	cout << "Usage: RGBDPlaneDetection <-o> color_image depth_image output_folder" << endl;
	cout << "-o: run MRF-optimization based plane refinement" << endl;
}

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		// printUsage();
		std::cerr << "invalid number of arguments, expected 1, got " << (argc - 1) << std::endl;
		return -1;
	}

  // parse arguments
	// bool run_mrf = string(argv[1]) == "-o" ? true : false;
	// string color_filename = run_mrf ? string(argv[2]) : string(argv[1]);
	// string depth_filename = run_mrf ? string(argv[3]) : string(argv[2]);
	// string output_folder = run_mrf ? string(argv[4]) : string(argv[3]);
	// string output_folder = string(argv[1]);
	// parse the minSupport int
	int minSupport = atoi(argv[1]);
	// check if it was a number
	if (minSupport <= 0) {
		cout << "Error: minSupport must be a positive number" << endl;
		return -1;
	}	
	
  // open ifstream with "/dev/stdin" in binary mode
	std::ifstream ifs("/dev/stdin", std::ios::binary);

	// read width, height from stdin
	int width, height;
	ifs.read(reinterpret_cast<char*>(&width), sizeof(width));
	ifs.read(reinterpret_cast<char*>(&height), sizeof(height));

	/* std::vector<ColorType> colors;
	for (int i = 0; i < width * height; i++)
	{
	  uint8_t r, g, b;
		ifs.read(reinterpret_cast<char*>(&r), sizeof(r));
		ifs.read(reinterpret_cast<char*>(&g), sizeof(g));
		ifs.read(reinterpret_cast<char*>(&b), sizeof(b));
		colors.push_back(std::array<uint8_t, 3>{
			b,
			g,
			r
		});
	}
	if (colors.size() != width * height) {
		std::cerr << "Error: number of colors does not match width and height: " << width << " " << height << " " << colors.size() << std::endl;
		return -1;
	} */

	std::vector<float> depths;
	float depth;
	for (int i = 0; i < width * height; i++)
	{
		ifs.read(reinterpret_cast<char*>(&depth), sizeof(depth));
		depths.push_back(-depth);
	}

  // std::cout << "read 1 " << width << " " << height << std::endl;
  PlaneDetection plane_detection(width, height, minSupport);
  // std::cout << "read 2 " << colors.size() << std::endl;
	// plane_detection.readColorImage(colors);
  // std::cout << "read 3 " << depths.size() << std::endl;
	plane_detection.readDepthImage(depths);
  // std::cout << "read 4" << std::endl;
	plane_detection.runPlaneDetection();
  // std::cout << "read 5" << std::endl;

	// const bool run_mrf = argc >= 3;
	const bool run_mrf = false;
	if (run_mrf)
	{
    // std::cout << "read 6" << std::endl;
		plane_detection.prepareForMRF();
    // std::cout << "read 7" << std::endl;
		runMRFOptimization(plane_detection);
    // std::cout << "read 8" << std::endl;
	}
	// int pos = color_filename.find_last_of("/\\");
	// string frame_name = color_filename.substr(pos + 1);
	// frame_name = frame_name.substr(0, frame_name.length() - 10);
	const string output_folder = "output";
	const string frame_name = "frame";
	plane_detection.writeOutputFiles(output_folder, frame_name, run_mrf);
	return 0;
}