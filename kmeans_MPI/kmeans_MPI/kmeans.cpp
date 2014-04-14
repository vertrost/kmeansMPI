#include <cmath>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <mpi.h>

using namespace std;

typedef vector<double> Point;
typedef vector<Point> Points;

// Gives random number in range [0..max_value]
unsigned int UniformRandom(unsigned int max_value) {
    unsigned int rnd = ((static_cast<unsigned int>(rand()) % 32768) << 17) |
                       ((static_cast<unsigned int>(rand()) % 32768) << 2) |
                       rand() % 4;
    return ((max_value + 1 == 0) ? rnd : rnd % (max_value + 1));
}

double Distance(const Point& point1, const Point& point2) {
    double distance_sqr = 0;
    int dimensions = point1.size();
    for (int i = 0; i < dimensions; ++i) {
        distance_sqr += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    return sqrt(distance_sqr);
}

int FindNearestCentroid(const Points& centroids, const Point& point) {
    double min_distance = Distance(point, centroids[0]);
    int centroid_index = 0;
    for (int i = 1; i < centroids.size(); ++i) {
        double distance = Distance(point, centroids[i]);
        if (distance < min_distance) {
            min_distance = distance;
            centroid_index = i;
        }
    }
    return centroid_index;
}

// Calculates new centroid position as mean of positions of 3 random centroids
Point GetRandomPosition(const Points& centroids) {
    int K = centroids.size();
    int c1 = rand() % K;
    int c2 = rand() % K;
    int c3 = rand() % K;
    int dimensions = centroids[0].size();
    Point new_position(dimensions);
    for (int d = 0; d < dimensions; ++d) {
        new_position[d] = (centroids[c1][d] + centroids[c2][d] + centroids[c3][d]) / 3;
    }
    return new_position;
}

vector<int> KMeans(double *_data, int K, int N, int D) {
    int data_size = N;
    int dimensions = D;
    vector<int> clusters(data_size);

    // Initialize centroids randomly at data points
    Points centroids(K);
	int *_centroids = (int*)malloc(K*D*sizeof(double));
	Points data;
	data.assign(N, Point(D));
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < D; j++)
			data[i][j] = _data[i*N + j];
	}

	if (rank == 0) {
		for (int i = 0; i < K; ++i) {
			centroids[i] = data[UniformRandom(data_size - 1)];
		}
	}
    
    bool converged = false;
    while (!converged) {
        converged = true;
        for (int i = 0; i < data_size; ++i) {
            int nearest_cluster = FindNearestCentroid(centroids, data[i]);
            if (clusters[i] != nearest_cluster) {
                clusters[i] = nearest_cluster;
                converged = false;
            }
        }
        if (converged) {
            break;
        }

        vector<int> clusters_sizes(K);
        centroids.assign(K, Point(dimensions));
        for (int i = 0; i < data_size; ++i) {
            for (int d = 0; d < dimensions; ++d) {
                centroids[clusters[i]][d] += data[i][d];
            }
            ++clusters_sizes[clusters[i]];
        }
        for (int i = 0; i < K; ++i) {
            if (clusters_sizes[i] != 0) {
                for (int d = 0; d < dimensions; ++d) {
                    centroids[i][d] /= clusters_sizes[i];
                }
            } else {
                centroids[i] = GetRandomPosition(centroids);
            }
        }
    }

    return clusters;
}

void WriteOutput(const vector<int>& clusters, ofstream& output) {
    for (int i = 0; i < clusters.size(); ++i) {
        output << clusters[i] << endl;
    }
}

int main(int argc , char** argv) {

	int rank, commsize, len;
	char host[MPI_MAX_PROCESSOR_NAME];
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); //rank of current processor
	MPI_Comm_size(MPI_COMM_WORLD, &commsize); //amount of all processors
	MPI_Get_processor_name(host, &len); //get hostname

	int K, N, D;
	Points data;
	double *_data;
	double *data_local;
	ofstream output;

	if (rank == 0) {
		if (argc != 4) {
			std::printf("Usage: %s number_of_clusters input_file output_file\n", argv[0]);
			return 1;
		}

		K = atoi(argv[1]);

		char* input_file = argv[2];
		ifstream input;
		input.open(input_file, ifstream::in);
		if (!input) {
			cerr << "Error: input file could not be opened" << endl;
			return 1;
		}

		input >> N >> D;
		_data = (double*)malloc(N * D *sizeof(double));
		for (int i = 0; i < N; ++i) {
			for (int d = 0; d < D; ++d) {
				double coord;
				input >> coord;
				_data[i*N + d] = coord;
			}
		}
		input.close();

		char* output_file = argv[3];
		output.open(output_file, ifstream::out);
		if (!output) {
			cerr << "Error: output file could not be opened" << endl;
			return 1;
		}
	}

	//Checking if N cannot be divided by commsize
	if (N % commsize != 0)
		MPI_Abort(MPI_COMM_WORLD, 1);
	int partsize = N / commsize;
	data_local = (double*)malloc(partsize*D*sizeof(double));

	// Broadcasting variables
	MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Bcast(_data, N * D, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
    srand(123); // for reproducible results

	// ================= start calcing =======================

	int data_size = N;
	int dimensions = D;
	vector<int> clusters(data_size);

	// Initialize centroids randomly at data points
	Points centroids(K);
	int *_centroids = (int*)malloc(K*D*sizeof(double));

	Points data;
	data.assign(N, Point(D));
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < D; j++)
			data[i][j] = _data[i*N + j];
	}

	MPI_Scatter(_data, N*D, MPI_DOUBLE,
		data_local, partsize*D, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		for (int i = 0; i < K; ++i) {
			centroids[i] = data[UniformRandom(data_size - 1)];
			for (int j = 0; j < D; ++j) {
				_centroids[i*N + j] = centroids[i][j];
			}
		}
	}
	//test

	MPI_Bcast(_centroids, K*D, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (rank > 0) {
		for (int i = 0; i < K; ++i) {
			for (int j = 0; j < D; ++j) {
				centroids[i][j] = _centroids[i*N + j];
			}
		}
	}

	//data ready, centroids ready

	bool converged = false;
	while (!converged) {
		converged = true;
		for (int i = 0; i < partsize; ++i) {
			//int nearest_cluster = FindNearestCentroid(centroids, data[i]);
			Point p;
			for (int j = 0; j < D; ++j) {
				p[i] = data_local[i*partsize + j];
			}
			int nearest_cluster = FindNearestCentroid(centroids, p); //stopped here
			if (clusters[i] != nearest_cluster) {
				clusters[i] = nearest_cluster;
				converged = false;
			}
		}
		if (converged) {
			break;
		}

		vector<int> clusters_sizes(K);
		centroids.assign(K, Point(dimensions));
		for (int i = 0; i < data_size; ++i) {
			for (int d = 0; d < dimensions; ++d) {
				centroids[clusters[i]][d] += data[i][d];
			}
			++clusters_sizes[clusters[i]];
		}
		for (int i = 0; i < K; ++i) {
			if (clusters_sizes[i] != 0) {
				for (int d = 0; d < dimensions; ++d) {
					centroids[i][d] /= clusters_sizes[i];
				}
			}
			else {
				centroids[i] = GetRandomPosition(centroids);
			}
		}
	}

    //vector<int> clusters = KMeans(data, K, N, D);

    //WriteOutput(clusters, output);
    output.close();

    return 0;
}
