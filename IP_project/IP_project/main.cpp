#include "../Common/memory.h"
#include "../Common/image3d.h"
#include "../Core/raw_io.h"
#include "../Core/raw_io_exception.h"
#include <memory>
#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <limits>
#include <fstream>
#include <ctime>
#include <omp.h>

template <class T>
std::unique_ptr<mc::image3d<T>> load_image(const std::string& path, const unsigned int w, const unsigned int h, const unsigned int d)
{
	using namespace mc;

	image3d<T>* pImg = nullptr;

	try {
		raw_io<T> io(path.c_str());
		io.setEndianType(raw_io<T>::EENDIAN_TYPE::BIG);
		pImg = io.read(w, h, d);
	}
	catch (raw_io_exception& e) {
		std::cerr << e.what() << std::endl;
	}

	return std::make_unique<image3d<T>>(*pImg);
}

int main()
{
	// read moving & target images for registration.
	std::unique_ptr<mc::image3d<short>> img1 = load_image<short>("volume1_512x512x56.raw", 512, 512, 56);
	std::unique_ptr<mc::image3d<short>> img2 = load_image<short>("volume2_512x512x58.raw", 512, 512, 58);

	/*
	*	EXAMPLES.
	* access data array :
	*	short** image_array = img1->data();
	* see more details at "../Common/image3d.h"
	*/

	// TODO #0 (optional) : Semi-isotropic image generation. (insert 4 slices for each)

	// TODO #1 : segmentation of lung region. (use thresholding, CCA)
	// allocate label image CCA.
	
	//img1
	std::unique_ptr<mc::image3d<unsigned int>> lmap{
		new mc::image3d<unsigned int>(img1->width(), img1->height(), img1->depth())
	};
	ZERO_VOLUME(unsigned int, lmap->data(), lmap->depth(), lmap->sizeSlice());

	//img1
	const int _w = lmap->width();
	const int _h = lmap->height();
	const int _d = lmap->depth();


	// Thresholding.
	omp_set_num_threads(4);
#pragma omp parallel for
	for (int i = 0; i < _d; ++i)
	for (int j = 0; j < _w*_h; ++j)
	{
		if (img1->data(i)[j] < -400) //-1024 ~ -401 를 foreground로 처리
		{
			lmap->data(i)[j] = 1;	// set foreground value (1).
		}
	}


	// CCA.
	std::clock_t start = std::clock();

	std::vector<unsigned int> l2p;	// label-2-parent mapper.
	l2p.push_back(0); //dummy for index convenience..

	unsigned int currentLabel = 0;
	// Run First Pass (generate label image with labels-2-parents mapping).
	for (int z = 0; z < _d; ++z) //depth 별로 돌면서
	for (int y = 0; y < _h; ++y)
	for (int x = 0; x < _w; ++x)
	{
		if (lmap->data(z)[x + y*_w] != 0) // if foreground.
		{
			unsigned int nLables[3] = { 0, 0, 0 };
			unsigned int cnt = 0;

			// seek backward connectivities. (3 when using 6-connectivity analysis)
			if ((z - 1 >= 0) && (lmap->get(x, y, z - 1) != 0)) {
				nLables[cnt++] = lmap->get(x, y, z - 1);
			}
			if ((y - 1 >= 0) && (lmap->get(x, y - 1, z) != 0)) {
				nLables[cnt++] = lmap->get(x, y - 1, z);
			}
			if ((x - 1 >= 0) && (lmap->get(x - 1, y, z) != 0)) {
				nLables[cnt++] = lmap->get(x - 1, y, z);
			}

			// neighbor assigned label exists.
			if (cnt > 0)
			{
				if (cnt == 1)
				{
					lmap->data(z)[x + y*_w] = nLables[0];
				}
				else
				{
					// find minimum neighbor.
					unsigned int nMin = std::min(nLables[0], nLables[1]);
					if (cnt == 3) nMin = std::min(nMin, nLables[2]);

					// assign minimum neighbor.
					lmap->data(z)[x + y*_w] = nMin;

					// change neighboring labels' mapping (might be redundant).
					for (int i = 0; i < cnt; ++i)
					{
						// this part can be optimized.
						if (l2p[nLables[i]] != l2p[nMin]) {
							////change l2p[nLables[i]] to l2p[nMin].
							for (auto& x : l2p)
							{
								if (x == l2p[nLables[i]])
									x = l2p[nMin];
							}
							//l2p[nLables[i]] = l2p[nMin];
						}
					}
				}
			}
			else
			{
				++currentLabel;
				l2p.push_back(currentLabel); // add currentLabel -> currentLabel mapping.
				lmap->data(z)[x + y*_w] = currentLabel;
			}
		}
	}
	
	// clean parent labels (just for clerity, w/o optimization).
	const auto lsz = l2p.size();
	unsigned int* check_tbl = new unsigned int[lsz];
	std::memset(check_tbl, 0, sizeof(unsigned int) * lsz);

	typedef std::pair<unsigned int, unsigned int> c2i; // count-to-index pair.
	std::vector<c2i> count_tbl;
	count_tbl.push_back(std::make_pair(std::numeric_limits<unsigned int>::max(), 0)); // set 0 label as superior priority.
	unsigned int number_of_labels = 0;
	for (auto& pl : l2p)
	{
		if (pl != 0)
		{
			if (check_tbl[pl] != 0)
			{
				pl = check_tbl[pl];
			}
			else
			{
				++number_of_labels;
				count_tbl.push_back(std::make_pair(0, number_of_labels)); // zero count.
				check_tbl[pl] = number_of_labels;
				pl = number_of_labels;
			}
		}
	}
	delete[](check_tbl);
	std::cout << "ccl obj count : " << number_of_labels << std::endl;
	// count volumes.
	for( int i=0; i<_d; ++i)
	for (int j = 0; j < _w*_h; ++j)
	{
		++count_tbl[l2p[lmap->data(i)[j]]].first;
	}
	
	// sort indices w.r.t. volume size.
	std::sort(count_tbl.begin(), count_tbl.end(), [](const c2i& lhs, const c2i& rhs) { return lhs.first < rhs.first; });

	// correct indices.
	std::vector<unsigned int> lut(count_tbl.size());
	for (int i = 0; i < lut.size(); ++i)
	{
		lut[count_tbl[i].second] = i;
	}
	
	// final mapping.
	std::unique_ptr<mc::image3d<unsigned int>> lmap2{
		new mc::image3d<unsigned int>(_w, _h, _d)
	};
	omp_set_num_threads(4);
#pragma omp parallel for
	// Run Second Pass (integration).
	for (int z = 0; z < _d; ++z)
	for (int y = 0; y < _h; ++y)
	for (int x = 0; x < _w; ++x)
	{
		lmap2->data(z)[x + y*_w] = lut[l2p[lmap->data(z)[x + y*_w]]];
	}
	// CCA done.
	double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "cca duration : " << duration << std::endl;

	// Test - export CCA label image.
	std::ofstream of("ccl_test.raw", std::ios::binary | std::ios::out);
	for (int i = 0; i < lmap2->depth(); ++i) {
		of.write(reinterpret_cast<char*>(lmap2->data(i)), sizeof(unsigned int) * lmap2->width() * lmap2->height());
	}
	of.close();// clean parent labels (just for clerity, w/o optimization).
	const auto lsz = l2p.size();
	unsigned int* check_tbl = new unsigned int[lsz];
	std::memset(check_tbl, 0, sizeof(unsigned int) * lsz);

	typedef std::pair<unsigned int, unsigned int> c2i; // count-to-index pair.
	std::vector<c2i> count_tbl;
	count_tbl.push_back(std::make_pair(std::numeric_limits<unsigned int>::max(), 0)); // set 0 label as superior priority.
	unsigned int number_of_labels = 0;
	for (auto& pl : l2p)
	{
		if (pl != 0)
		{
			if (check_tbl[pl] != 0)
			{
				pl = check_tbl[pl];
			}
			else
			{
				++number_of_labels;
				count_tbl.push_back(std::make_pair(0, number_of_labels)); // zero count.
				check_tbl[pl] = number_of_labels;
				pl = number_of_labels;
			}
		}
	}
	delete[](check_tbl);
	std::cout << "ccl obj count : " << number_of_labels << std::endl;
	// count volumes.
	for( int i=0; i<_d; ++i)
	for (int j = 0; j < _w*_h; ++j)
	{
		++count_tbl[l2p[lmap->data(i)[j]]].first;
	}
	
	// sort indices w.r.t. volume size.
	std::sort(count_tbl.begin(), count_tbl.end(), [](const c2i& lhs, const c2i& rhs) { return lhs.first < rhs.first; });

	// correct indices.
	std::vector<unsigned int> lut(count_tbl.size());
	for (int i = 0; i < lut.size(); ++i)
	{
		lut[count_tbl[i].second] = i;
	}
	
	// final mapping.
	std::unique_ptr<mc::image3d<unsigned int>> lmap2{
		new mc::image3d<unsigned int>(_w, _h, _d)
	};
	omp_set_num_threads(4);
#pragma omp parallel for
	// Run Second Pass (integration).
	for (int z = 0; z < _d; ++z)
	for (int y = 0; y < _h; ++y)
	for (int x = 0; x < _w; ++x)
	{
		lmap2->data(z)[x + y*_w] = lut[l2p[lmap->data(z)[x + y*_w]]];
	}
	// CCA done.
	double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "cca duration : " << duration << std::endl;

	// Test - export CCA label image.
	std::ofstream of("ccl_test.raw", std::ios::binary | std::ios::out);
	for (int i = 0; i < lmap2->depth(); ++i) {
		of.write(reinterpret_cast<char*>(lmap2->data(i)), sizeof(unsigned int) * lmap2->width() * lmap2->height());
	}
	of.close();

	// TODO #1-1 : Initial transformation parameter calculation.

	// TODO #2 : edge extraction for both images.

	// TODO #3 : Distance transformation.

	// TODO #4 : Perform iterative REGISTRATION.

	// TODO #5 : Transform moving (floating) image with estimated transformation parameter & generate subtraction image.

	// TODO #6 : store subtraction image (visual purpose).

	// TODO :
	// perform Intensity-based registration (using similarity measure metric with original intensities).
	// Do #[4, 5, 6] and compare with surface-based method.

	return EXIT_SUCCESS;
}

void CCA(std::unique_ptr<mc::image3d<short>> img) {

	std::unique_ptr<mc::image3d<unsigned int>> lmap{
		new mc::image3d<unsigned int>(img->width(), img->height(), img->depth())
	};
	ZERO_VOLUME(unsigned int, lmap->data(), lmap->depth(), lmap->sizeSlice());


	const int _w = lmap->width();
	const int _h = lmap->height();
	const int _d = lmap->depth();

	//Thresholding
	omp_set_num_threads(4);
#pragma omp parallel for
	for (int i = 0; i < _d; ++i)
		for (int j = 0; j < _w * _h; ++j)
		{
			if (img->data(i)[j] < -400) //-1024 ~ -401 를 foreground로 처리
			{
				lmap->data(i)[j] = 1;	// set foreground value (1).
			}
		}

	// CCA.
	std::clock_t start = std::clock();

	std::vector<unsigned int> l2p;	// label-2-parent mapper.
	l2p.push_back(0); //dummy for index convenience..

	unsigned int currentLabel = 0;
	// Run First Pass (generate label image with labels-2-parents mapping).
	for (int z = 0; z < _d; ++z) //depth 별로 돌면서
		for (int y = 0; y < _h; ++y)
			for (int x = 0; x < _w; ++x)
			{
				if (lmap->data(z)[x + y * _w] != 0) // if foreground.
				{
					unsigned int nLables[3] = { 0, 0, 0 };
					unsigned int cnt = 0;

					// seek backward connectivities. (3 when using 6-connectivity analysis)
					if ((z - 1 >= 0) && (lmap->get(x, y, z - 1) != 0)) {
						nLables[cnt++] = lmap->get(x, y, z - 1);
					}
					if ((y - 1 >= 0) && (lmap->get(x, y - 1, z) != 0)) {
						nLables[cnt++] = lmap->get(x, y - 1, z);
					}
					if ((x - 1 >= 0) && (lmap->get(x - 1, y, z) != 0)) {
						nLables[cnt++] = lmap->get(x - 1, y, z);
					}

					// neighbor assigned label exists.
					if (cnt > 0)
					{
						if (cnt == 1)
						{
							lmap->data(z)[x + y * _w] = nLables[0];
						}
						else
						{
							// find minimum neighbor.
							unsigned int nMin = std::min(nLables[0], nLables[1]);
							if (cnt == 3) nMin = std::min(nMin, nLables[2]);

							// assign minimum neighbor.
							lmap->data(z)[x + y * _w] = nMin;

							// change neighboring labels' mapping (might be redundant).
							for (int i = 0; i < cnt; ++i)
							{
								// this part can be optimized.
								if (l2p[nLables[i]] != l2p[nMin]) {
									////change l2p[nLables[i]] to l2p[nMin].
									for (auto& x : l2p)
									{
										if (x == l2p[nLables[i]])
											x = l2p[nMin];
									}
									//l2p[nLables[i]] = l2p[nMin];
								}
							}
						}
					}
					else
					{
						++currentLabel;
						l2p.push_back(currentLabel); // add currentLabel -> currentLabel mapping.
						lmap->data(z)[x + y * _w] = currentLabel;
					}
				}
			}

	// clean parent labels (just for clerity, w/o optimization).
	const auto lsz = l2p.size();
	unsigned int* check_tbl = new unsigned int[lsz];
	std::memset(check_tbl, 0, sizeof(unsigned int) * lsz);

	typedef std::pair<unsigned int, unsigned int> c2i; // count-to-index pair.
	std::vector<c2i> count_tbl;
	count_tbl.push_back(std::make_pair(std::numeric_limits<unsigned int>::max(), 0)); // set 0 label as superior priority.
	unsigned int number_of_labels = 0;
	for (auto& pl : l2p)
	{
		if (pl != 0)
		{
			if (check_tbl[pl] != 0)
			{
				pl = check_tbl[pl];
			}
			else
			{
				++number_of_labels;
				count_tbl.push_back(std::make_pair(0, number_of_labels)); // zero count.
				check_tbl[pl] = number_of_labels;
				pl = number_of_labels;
			}
		}
	}
	delete[](check_tbl);
	std::cout << "ccl obj count : " << number_of_labels << std::endl;
	// count volumes.
	for( int i=0; i<_d; ++i)
	for (int j = 0; j < _w*_h; ++j)
	{
		++count_tbl[l2p[lmap->data(i)[j]]].first;
	}
	
	// sort indices w.r.t. volume size.
	std::sort(count_tbl.begin(), count_tbl.end(), [](const c2i& lhs, const c2i& rhs) { return lhs.first < rhs.first; });

	// correct indices.
	std::vector<unsigned int> lut(count_tbl.size());
	for (int i = 0; i < lut.size(); ++i)
	{
		lut[count_tbl[i].second] = i;
	}
	
	// final mapping.
	std::unique_ptr<mc::image3d<unsigned int>> lmap2{
		new mc::image3d<unsigned int>(_w, _h, _d)
	};
	omp_set_num_threads(4);
#pragma omp parallel for
	// Run Second Pass (integration).
	for (int z = 0; z < _d; ++z)
	for (int y = 0; y < _h; ++y)
	for (int x = 0; x < _w; ++x)
	{
		lmap2->data(z)[x + y*_w] = lut[l2p[lmap->data(z)[x + y*_w]]];
	}
	// CCA done.
	double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "cca duration : " << duration << std::endl;

	// Test - export CCA label image.
	std::ofstream of("ccl_test.raw", std::ios::binary | std::ios::out);
	for (int i = 0; i < lmap2->depth(); ++i) {
		of.write(reinterpret_cast<char*>(lmap2->data(i)), sizeof(unsigned int) * lmap2->width() * lmap2->height());
	}
	of.close();

}