#include "plane_detection.h"
#include <stdint.h>
#include <iomanip> // output double value precision

//

constexpr float scaleFactor = 1000.f; // scale is in mm

//

PlaneDetection::PlaneDetection(int w, int h, int minSupport)
{
	cloud.w = w;
	cloud.h = h;
	cloud.vertices.resize(cloud.h * cloud.w);

  // maxStep(100000), minSupport(3000),
	// windowWidth(10), windowHeight(10),
	// doRefine(true), erodeType(ERODE_ALL_BORDER),
	// dirtyBlkMbship(true), drawCoarseBorder(false)
	// plane_filter.maxStep = 1000000;
	plane_filter.minSupport = minSupport;
	plane_filter.windowWidth = 4;
	plane_filter.windowHeight = 4;
	// plane_filter.windowWidth = 8;
	// plane_filter.windowHeight = 8;
	// plane_filter.dirtyBlkMbship = false;
	// plane_filter.erodeType = ahc::ERODE_NONE;
}

PlaneDetection::~PlaneDetection()
{
	cloud.vertices.clear();
	seg_img_.release();
	opt_seg_img_.release();
	color_img_.release();
	opt_membership_img_.release();
	pixel_boundary_flags_.clear();
	pixel_grayval_.clear();
	plane_colors_.clear();
	plane_pixel_nums_.clear();
	opt_plane_pixel_nums_.clear();
	sum_stats_.clear();
	opt_sum_stats_.clear();
}

// Temporarily don't need it since we set intrinsic parameters as constant values in the code.
//bool PlaneDetection::readIntrinsicParameterFile(string filename)
//{
//	ifstream readin(filename, ios::in);
//	if (readin.fail() || readin.eof())
//	{
//		cout << "WARNING: Cannot read intrinsics file " << filename << endl;
//		return false;
//	}
//	string target_str = "m_calibrationDepthIntrinsic";
//	string str_line, str, str_dummy;
//	double dummy;
//	bool read_success = false;
//	while (!readin.eof() && !readin.fail())
//	{
//		getline(readin, str_line);
//		if (readin.eof())
//			break;
//		istringstream iss(str_line);
//		iss >> str;
//		if (str == "m_depthWidth")
//			iss >> str_dummy >> width_;
//		else if (str == "m_depthHeight")
//			iss >> str_dummy >> height_;
//		else if (str == "m_calibrationDepthIntrinsic")
//		{
//			iss >> str_dummy >> fx_ >> dummy >> cx_ >> dummy >> dummy >> fy_ >> cy_;
//			read_success = true;
//			break;
//		}
//	}
//	readin.close();
//	if (read_success)
//	{
//		cloud.vertices.resize(height_ * width_);
//		cloud.w = width_;
//		cloud.h = height_;
//	}
//	return read_success;
//}

bool PlaneDetection::readColorImage(string filename)
{
	color_img_ = cv::imread(filename, cv::IMREAD_COLOR);
	if (color_img_.empty() || color_img_.depth() != CV_8U)
	{
		cerr << "ERROR: cannot read color image. No such a file, or the image format is not 8UC3" << endl;
		return false;
	}
	return true;
}
bool PlaneDetection::readColorImage(std::vector<ColorType> &colors)
{
	// color_img_ = cv::imread(filename, cv::IMREAD_COLOR);
	// if (color_img_.empty() || color_img_.depth() != CV_8U)
	// {
	// 	cerr << "ERROR: cannot read color image. No such a file, or the image format is not 8UC3" << endl;
	// 	return false;
	// }
	// note: OpenCV uses BGR
	color_img_ = cv::Mat(cloud.h, cloud.w, CV_8UC3, colors.data());
	return true;
}

bool PlaneDetection::readDepthImage(string filename)
{
	cv::Mat depth_img = cv::imread(filename, cv::IMREAD_ANYDEPTH);
	if (depth_img.empty() || depth_img.depth() != CV_16U)
	{
		cerr << "WARNING: cannot read depth image. No such a file, or the image format is not 16UC1" << endl;
		return false;
	}
	int rows = depth_img.rows, cols = depth_img.cols;
	int vertex_idx = 0;
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			double z = (double)(depth_img.at<unsigned short>(i, j)) / kScaleFactor;
			if (_isnan(z))
			{
				cloud.vertices[vertex_idx++] = VertexType(0, 0, z);
				continue;
			}
			const float kFx = (double)cloud.w;
			const float kFy = (double)cloud.h;
			const float kCx = (double)cloud.w / 2.0;
			const float kCy = (double)cloud.h / 2.0;

			double x = ((double)j - kCx) * z / kFx;
			double y = ((double)i - kCy) * z / kFy;
			cloud.vertices[vertex_idx++] = VertexType(x, y, z);
		}
	}
	return true;
}
bool PlaneDetection::readDepthImage(std::vector<float> &depths)
{
	// cv::Mat depth_img = cv::imread(filename, cv::IMREAD_ANYDEPTH);
	// if (depth_img.empty() || depth_img.depth() != CV_16U)
	// {
	// 	cerr << "WARNING: cannot read depth image. No such a file, or the image format is not 16UC1" << endl;
	// 	return false;
	// }
	// int rows = depth_img.rows, cols = depth_img.cols;
	int rows = cloud.h, cols = cloud.w;
	int vertex_idx = 0;
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			// double z = (double)(depth_img.at<unsigned short>(i, j)) / kScaleFactor;
			double z = (double)(depths[vertex_idx] * scaleFactor);
			if (_isnan(z))
			{
				cloud.vertices[vertex_idx++] = VertexType(0, 0, z);
				continue;
			}
			const float kFx = (double)cloud.w;
			const float kFy = (double)cloud.h;
			const float kCx = (double)cloud.w / 2.0;
			const float kCy = (double)cloud.h / 2.0;

			double x = ((double)j - kCx) * z / kFx;
			double y = ((double)i - kCy) * z / kFy;
			cloud.vertices[vertex_idx++] = VertexType(x, y, z);
		}
	}
	return true;
}

bool PlaneDetection::runPlaneDetection()
{
	seg_img_ = cv::Mat(cloud.h, cloud.w, CV_8UC3);
	plane_filter.run(&cloud, &plane_vertices_, &seg_img_);
	plane_num_ = (int)plane_vertices_.size();

	// Here we set the plane index of a pixel which does NOT belong to any plane as #planes.
	// This is for using MRF optimization later.
	for (int row = 0; row < cloud.h; ++row)
		for (int col = 0; col < cloud.w; ++col)
			if (plane_filter.membershipImg.at<int>(row, col) < 0)
				plane_filter.membershipImg.at<int>(row, col) = plane_num_;
	return true;
}

void PlaneDetection::prepareForMRF()
{
	opt_seg_img_ = cv::Mat(cloud.h, cloud.w, CV_8UC3);
	opt_membership_img_ = cv::Mat(cloud.h, cloud.w, CV_32SC1);
	pixel_boundary_flags_.resize(cloud.w * cloud.h, false);
	pixel_grayval_.resize(cloud.w * cloud.h, 0);

	cv::Mat& mat_label = plane_filter.membershipImg;
	for (int row = 0; row < cloud.h; ++row)
	{
		for (int col = 0; col < cloud.w; ++col)
		{
			pixel_grayval_[row * cloud.w + col] = RGB2Gray(row, col);
			int label = mat_label.at<int>(row, col);
			if ((row - 1 >= 0 && mat_label.at<int>(row - 1, col) != label)
				|| (row + 1 < cloud.h && mat_label.at<int>(row + 1, col) != label)
				|| (col - 1 >= 0 && mat_label.at<int>(row, col - 1) != label)
				|| (col + 1 < cloud.w && mat_label.at<int>(row, col + 1) != label))
			{
				// Pixels in a fixed range near the boundary pixel are also regarded as boundary pixels
				for (int x = max(row - kNeighborRange, 0); x < min(cloud.h, row + kNeighborRange); ++x)
				{
					for (int y = max(col - kNeighborRange, 0); y < min(cloud.w, col + kNeighborRange); ++y)
					{
						// If a pixel is not on any plane, then it is not a boundary pixel.
						if (mat_label.at<int>(x, y) == plane_num_)
							continue;
						pixel_boundary_flags_[x * cloud.w + y] = true;
					}
				}
			}
		}
	}

	for (int pidx = 0; pidx < plane_num_; ++pidx)
	{
		int vidx = plane_vertices_[pidx][0];
		cv::Vec3b c = seg_img_.at<cv::Vec3b>(vidx / cloud.w, vidx % cloud.w);
		plane_colors_.push_back(c);
	}
	plane_colors_.push_back(cv::Vec3b(0,0,0)); // black for pixels not in any plane
}

// Note: input filename_prefix is like '/rgbd-image-folder-path/frame-XXXXXX'
void PlaneDetection::writeOutputFiles(string output_folder, string frame_name, bool run_mrf)
{
	computePlaneSumStats(run_mrf);

	/* if (output_folder.back() != '\\' && output_folder.back() != '/')
		output_folder += "/";	
	string filename_prefix = output_folder + frame_name + "-plane";
	cv::imwrite(filename_prefix + ".png", seg_img_);
	writePlaneLabelFile(filename_prefix + "-label.txt");
	writePlaneDataFile(filename_prefix + "-data.txt");
	if (run_mrf)
	{
		cv::imwrite(filename_prefix + "-opt.png", opt_seg_img_);
		writePlaneLabelFile(filename_prefix + "-label-opt.txt", run_mrf);
		writePlaneDataFile(filename_prefix + "-data-opt.txt", run_mrf);
	} */
	
}
void PlaneDetection::writePlaneLabelFile(string filename, bool run_mrf /* = false */)
{
	ofstream out(filename, ios::out);
	out << plane_num_ << endl;
	if (plane_num_ == 0)
	{
		out.close();
		return;
	}
	for (int row = 0; row < cloud.h; ++row)
	{
		for (int col = 0; col < cloud.w; ++col)
		{
			int label = run_mrf ? opt_membership_img_.at<int>(row, col) : plane_filter.membershipImg.at<int>(row, col);
			out << label << " ";
		}
		out << endl;
	}
	out.close();
}

void PlaneDetection::writePlaneDataFile(string filename, bool run_mrf /* = false */)
{
	ofstream out(filename, ios::out);
	out << "#plane_index number_of_points_on_the_plane plane_color_in_png_image(1x3) plane_normal(1x3) plane_center(1x3) "
		<< "sx sy sz sxx syy szz sxy syz sxz" << endl;

	for (int pidx = 0; pidx < plane_num_; ++pidx)
	{
		out << pidx << " ";
		if (!run_mrf)
			out << plane_pixel_nums_[pidx] << " ";
		else
			out << opt_plane_pixel_nums_[pidx] << " ";

		// Plane color in output image
		int vidx = plane_vertices_[pidx][0];
		cv::Vec3b c = seg_img_.at<cv::Vec3b>(vidx / cloud.w, vidx % cloud.w);
		out << int(c.val[2]) << " " << int(c.val[1]) << " "<< int(c.val[0]) << " "; // OpenCV uses BGR by default

		// Plane normal and center
		int new_pidx = pid_to_extractedpid[pidx];
		for (int i = 0; i < 3; ++i)
			out << plane_filter.extractedPlanes[new_pidx]->normal[i] << " ";
		for (int i = 0; i < 3; ++i)
			out << plane_filter.extractedPlanes[new_pidx]->center[i] << " ";

		// Sum of all points on the plane
		if (run_mrf)
		{
			out << opt_sum_stats_[pidx].sx << std::setprecision(8) << " " 
				<< opt_sum_stats_[pidx].sy << std::setprecision(8) << " " 
				<< opt_sum_stats_[pidx].sz << std::setprecision(8) << " " 
				<< opt_sum_stats_[pidx].sxx << std::setprecision(8) << " "
				<< opt_sum_stats_[pidx].syy << std::setprecision(8) << " "
				<< opt_sum_stats_[pidx].szz << std::setprecision(8) << " "
				<< opt_sum_stats_[pidx].sxy << std::setprecision(8) << " "
				<< opt_sum_stats_[pidx].syz << std::setprecision(8) << " "
				<< opt_sum_stats_[pidx].sxz << std::setprecision(8) << endl;
		}
		else
		{
			out << sum_stats_[pidx].sx << std::setprecision(8) << " " 
				<< sum_stats_[pidx].sy << std::setprecision(8) << " " 
				<< sum_stats_[pidx].sz << std::setprecision(8) << " " 
				<< sum_stats_[pidx].sxx << std::setprecision(8) << " "
				<< sum_stats_[pidx].syy << std::setprecision(8) << " "
				<< sum_stats_[pidx].szz << std::setprecision(8) << " "
				<< sum_stats_[pidx].sxy << std::setprecision(8) << " "
				<< sum_stats_[pidx].syz << std::setprecision(8) << " "
				<< sum_stats_[pidx].sxz << std::setprecision(8) << endl;
		}

		// NOTE: the plane-sum parameters computed from AHC code seems different from that computed from points belonging to planes shown above.
		// Seems there is a plane refinement step in AHC code so points belonging to each plane are slightly changed.
		//ahc::PlaneSeg::Stats& stat = plane_filter.extractedPlanes[pidx]->stats;
		//cerr << stat.sx << " " << stat.sy << " " << stat.sz << " " << stat.sxx << " "<< stat.syy << " "<< stat.szz << " "<< stat.sxy << " "<< stat.syz << " "<< stat.sxz << endl;
	}
	out.close();
}

void PlaneDetection::computePlaneSumStats(bool run_mrf /* = false */)
{
	sum_stats_.resize(plane_num_);
	for (int pidx = 0; pidx < plane_num_; ++pidx)
	{
		for (int i = 0; i < plane_vertices_[pidx].size(); ++i)
		{
			int vidx = plane_vertices_[pidx][i];
			const VertexType& v = cloud.vertices[vidx];
			sum_stats_[pidx].sx += v[0];		 sum_stats_[pidx].sy += v[1];		  sum_stats_[pidx].sz += v[2];
			sum_stats_[pidx].sxx += v[0] * v[0]; sum_stats_[pidx].syy += v[1] * v[1]; sum_stats_[pidx].szz += v[2] * v[2];
			sum_stats_[pidx].sxy += v[0] * v[1]; sum_stats_[pidx].syz += v[1] * v[2]; sum_stats_[pidx].sxz += v[0] * v[2];
		}
		plane_pixel_nums_.push_back(int(plane_vertices_[pidx].size()));
	}
	for (int pidx = 0; pidx < plane_num_; ++pidx)
	{
		int num = plane_pixel_nums_[pidx];
		sum_stats_[pidx].sx /= num;		sum_stats_[pidx].sy /= num;		sum_stats_[pidx].sz /= num;
		sum_stats_[pidx].sxx /= num;	sum_stats_[pidx].syy /= num;	sum_stats_[pidx].szz /= num;
		sum_stats_[pidx].sxy /= num;	sum_stats_[pidx].syz /= num;	sum_stats_[pidx].sxz /= num;
	}
	// Note that the order of extracted planes in `plane_filter.extractedPlanes` is DIFFERENT from
	// the plane order in `plane_vertices_` after running plane detection function `plane_filter.run()`.
	// So here we compute a mapping between these two types of plane indices by comparing plane centers.
	vector<double> sx(plane_num_), sy(plane_num_), sz(plane_num_);
	for (int i = 0; i < plane_filter.extractedPlanes.size(); ++i)
	{
		sx[i] = plane_filter.extractedPlanes[i]->stats.sx / plane_filter.extractedPlanes[i]->stats.N;
		sy[i] = plane_filter.extractedPlanes[i]->stats.sy / plane_filter.extractedPlanes[i]->stats.N;
		sz[i] = plane_filter.extractedPlanes[i]->stats.sz / plane_filter.extractedPlanes[i]->stats.N;
	}
	extractedpid_to_pid.clear();
	pid_to_extractedpid.clear();
	// If two planes' centers are closest, then the two planes are corresponding to each other.
	for (int i = 0; i < plane_num_; ++i)
	{
		double min_dis = 1000000;
		int min_idx = -1;
		for (int j = 0; j < plane_num_; ++j)
		{
			double a = sum_stats_[i].sx - sx[j], b = sum_stats_[i].sy - sy[j], c = sum_stats_[i].sz - sz[j];
			double dis = a * a + b * b + c * c;
			if (dis < min_dis)
			{
				min_dis = dis;
				min_idx = j;
			}
		}
		if (extractedpid_to_pid.find(min_idx) != extractedpid_to_pid.end())
		{
			cerr << "   WARNING: a mapping already exists for extracted plane " << min_idx << ":" << extractedpid_to_pid[min_idx] << " -> " << min_idx << endl;
		}
		pid_to_extractedpid[i] = min_idx;
		extractedpid_to_pid[min_idx] = i;
	}
	if (run_mrf)
	{
		opt_sum_stats_.resize(plane_num_);
		opt_plane_pixel_nums_.resize(plane_num_, 0);
		for (int row = 0; row < cloud.h; ++row)
		{
			for (int col = 0; col < cloud.w; ++col)
			{
				int label = opt_membership_img_.at<int>(row, col); // plane label each pixel belongs to
				if (label != plane_num_) // pixel belongs to some plane
				{
					opt_plane_pixel_nums_[label]++;
					int vidx = row * cloud.w + col;
					const VertexType& v = cloud.vertices[vidx];
					opt_sum_stats_[label].sx += v[0];		  opt_sum_stats_[label].sy += v[1];		    opt_sum_stats_[label].sz += v[2];
					opt_sum_stats_[label].sxx += v[0] * v[0]; opt_sum_stats_[label].syy += v[1] * v[1]; opt_sum_stats_[label].szz += v[2] * v[2];
					opt_sum_stats_[label].sxy += v[0] * v[1]; opt_sum_stats_[label].syz += v[1] * v[2]; opt_sum_stats_[label].sxz += v[0] * v[2];
				}
			}
		}
		for (int pidx = 0; pidx < plane_num_; ++pidx)
		{
			int num = opt_plane_pixel_nums_[pidx];
			opt_sum_stats_[pidx].sx /= num;		opt_sum_stats_[pidx].sy /= num;		opt_sum_stats_[pidx].sz /= num;
			opt_sum_stats_[pidx].sxx /= num;	opt_sum_stats_[pidx].syy /= num;	opt_sum_stats_[pidx].szz /= num;
			opt_sum_stats_[pidx].sxy /= num;	opt_sum_stats_[pidx].syz /= num;	opt_sum_stats_[pidx].sxz /= num;
		}
	}

	//--------------------------------------------------------------
	// Only for debug. It doesn't influence the plane detection.
	uint32_t numPlanes = plane_num_;
  cout.write(reinterpret_cast<char*>(&numPlanes), sizeof(numPlanes));

	std::vector<int> planeIndices(cloud.w * cloud.h);
	for (int pidx = 0; pidx < plane_num_; ++pidx)
	{
    auto &plane = plane_filter.extractedPlanes[pidx];

		double w = 0;
		//for (int j = 0; j < 3; ++j)
		//	w -= plane->normal[j] * plane->center[j];
		w -= plane->normal[0] * sum_stats_[pidx].sx;
		w -= plane->normal[1] * sum_stats_[pidx].sy;
		w -= plane->normal[2] * sum_stats_[pidx].sz;
		double sum = 0;
		for (int i = 0; i < plane_vertices_[pidx].size(); ++i)
		{
			int vidx = plane_vertices_[pidx][i];
			const VertexType& v = cloud.vertices[vidx];
			double dis = w;
			for (int j = 0; j < 3; ++j)
				dis += v[j] * plane->normal[j];
			dis /= scaleFactor;
			sum += dis * dis;

			planeIndices[vidx] = pidx;
		}
		sum /= plane_vertices_[pidx].size();

    float normal[3] = {
      (float)plane->normal[0],
			(float)plane->normal[1],
			(float)plane->normal[2]
		};
    cout.write(reinterpret_cast<char*>(normal), sizeof(normal));
    float center[3] = {
			(float)plane->center[0] / scaleFactor,
			(float)plane->center[1] / scaleFactor,
			(float)plane->center[2] / scaleFactor
		};
    cout.write(reinterpret_cast<char*>(center), sizeof(center));
		uint32_t numVertices = plane_vertices_[pidx].size();
		cout.write(reinterpret_cast<char*>(&numVertices), sizeof(numVertices));
		float distanceSquaredF = (float)sum;
		cout.write(reinterpret_cast<char*>(&distanceSquaredF), sizeof(distanceSquaredF));
		
		// cout << "Plane " << pidx <<
		//   " normal: " << plane->normal[0] << " " << plane->normal[1] << " " << plane->normal[2] <<
		// 	" center: " <<
		// 	  (plane->center[0] / scaleFactor) << " " <<
		// 		(plane->center[1] / scaleFactor) << " " <<
		// 		(-plane->center[2] / scaleFactor) << " " <<
		// 	" numVertices: " << plane_vertices_[pidx].size() << " " <<
		//   " distanceSquared: " << sum << endl;
	}

	// std::cout << "Plane indices: " << planeIndices.size() << std::endl;
	// for (int i = 0; i < 32; i++) {
	// 	std::cout << planeIndices[i] << " ";
	// }
	// std::cout << std::endl;

  cout.write(reinterpret_cast<char*>(planeIndices.data()), planeIndices.size() * sizeof(planeIndices[0]));
}
