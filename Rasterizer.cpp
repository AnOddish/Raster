#include <sycl/sycl.hpp>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <numbers>
#include <random>

constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;

//The basic operators for 2D vector maths
struct Vec2 {
    float i;
    float j;

    Vec2(float X, float Y) : i(X), j(Y) {}

    Vec2 operator+(const Vec2& other) {
        return { i + other.i, j + other.j };
    }

    Vec2 operator-(Vec2 other) {
        return { i - other.i, j - other.j };
    }

    Vec2 operator*(const float& scalar) {
        return { i * scalar, j * scalar };
    }
};

// The basic operators for 3D vector maths
struct Vec3 {
    float i;
    float j;
    float k;

    Vec3(float X, float Y, float Z) : i(X), j(Y), k(Z) {}

    // adding 2 vectors
    Vec3 operator+(const Vec3& other) const {
        return { i + other.i, j + other.j, k + other.k };
    }

    Vec3 operator-(const Vec3& other) {
        return { i - other.i, j - other.j, k - other.k };
    }

    // multiplying a vector by a scalar value
    Vec3 operator*(const float& scalar) const {
        return { i * scalar, j * scalar, k * scalar };
    }
};

// the dot product of two 3D vectors
float dot(const Vec3& v1, const Vec3& v2) {
    return v1.i * v2.i + v1.j * v2.j + v1.k * v2.k;
}

// this is a normalised vector where W should always be 1
struct Vec4 {
    float i;
    float j;
    float k;
    float w;

    Vec4(float X, float Y, float Z, float W) : i(X), j(Y), k(Z), w(W) {}
    Vec4(Vec3 base, float W) : i(base.i), j(base.j), k(base.k), w(W) {}

    Vec4 operator/(float scalar) {
        return { i / scalar, j / scalar, k / scalar, w / scalar };
    }
};

// this is a 4x4 matrix
struct Matrix {
    float data[16];

    // initialise an array 
    Matrix() {
        for (int i = 0; i < 16; i++) {
            data[i] = 0;
        }
        data[0] = 1;
        data[1 * 4 + 1] = 1;
        data[2 * 4 + 2] = 1;
        data[3 * 4 + 3] = 1;
    }

    float& operator()(size_t x, size_t y) {
        return data[y * 4 + x];
    }

    Matrix operator*(const Matrix& other) const {
        //Create a matrix full of 0s
        Matrix result;
        for (int i = 0; i < 16; i++) {
            result.data[i] = 0;
        }

        // a quick a dirty 4x4 x 4x4 matrix multiplication algorithm
        // its not the point here
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                float sum = 0.0f;
                for (int k = 0; k < 4; k++) {
                    sum += data[row * 4 + k] * other.data[k * 4 + col];
                }
                result.data[row * 4 + col] = sum;
            }
        }
        return result;
    }

    Vec4 operator*(const Vec4& vec) const {
        float vals[4] = { vec.i, vec.j, vec.k, vec.w };
        float out[4] = {};
        for (int row = 0; row < 4; row++)
            for (int col = 0; col < 4; col++)
                out[row] += data[row * 4 + col] * vals[col];

        Vec4 vOut = { out[0], out[1], out[2], out[3] };
        return vOut;
    }
};

//Defines the structure of a vertex
struct Vertex {
    Vec3 position;
    Vec3 colour;

    // Some constructors
    Vertex(Vec3 pos) : position(pos), colour({ 1.f, 1.f, 1.f }) {};
    Vertex(Vec3 pos, Vec3 col) : position(pos), colour(col) {};
};

struct Tri {
    Vec3 positions[3];
    Vec3 colours[3];
};

// This function is used to calculate the denominator of the edges weights used to simplify the maths when converting to barycentric coordinates
float edge(const Vec3& a, const Vec3& b, const Vec3& c) {
    return (b.i - a.i) * (c.j - a.j) - (b.j - a.j) * (c.i - a.i);
}

Matrix GetViewMatrix(Vec3 right, Vec3 up, Vec3 front, Vec3 position) {
    //This matrix is used to set the position of the camera and its rotation

    Matrix view;
    view(0, 0) = right.i;
    view(1, 0) = right.j;
    view(2, 0) = right.k;

    view(0, 1) = up.i;
    view(1, 1) = up.j;
    view(2, 1) = up.k;

    view(0, 2) = front.i;
    view(1, 2) = front.j;
    view(2, 2) = front.k;

    view(0, 3) = -dot(right, position);
    view(1, 3) = -dot(up, position);
    view(2, 3) = -dot(front, position);

    return view;
}

// fov in degrees
Matrix GetPerspectiveMatrix(float fov, float near, float far) {
    // calculates the perspective matrix used to convert from 3D coordinates into screen space
    float f = 1 / std::tan((fov * (std::numbers::pi / 180.f)) / 2.f);
    float aspectRatio = static_cast<float>(WIDTH) / static_cast<float>(HEIGHT);
    Matrix perp;
    perp(0, 0) = f / aspectRatio;
    perp(1, 1) = f;
    perp(2, 2) = (far + near) / (near - far);
    perp(3, 2) = (2 * far * near) / (near - far);
    perp(2, 3) = -1.f;
    perp(3, 3) = 0;

    return perp;
}

// saves the resultant image to bit map file
void WriteImageToFile(std::string fileName, sycl::host_accessor<Vec3, 2> imageBuffer) {
    std::ofstream file(fileName + ".bmp", std::ios::binary);
    if (!file) {
        throw std::runtime_error("Can't create file");
    }

    const uint32_t fileHeaderSize = 14;
    const uint32_t dibHeaderSize = 40;
    const uint32_t dataSize = WIDTH * HEIGHT * 4;
    const uint32_t totalFileSize = fileHeaderSize + dibHeaderSize + dataSize;
    const uint32_t dataOffset = fileHeaderSize + dibHeaderSize;

    // BMP Header ("BM")
    file.put('B').put('M');

    // Write header fields
    file.write(reinterpret_cast<const char*>(&totalFileSize), 4); // file size
    uint32_t reserved = 0;
    file.write(reinterpret_cast<const char*>(&reserved), 4);      // reserved
    file.write(reinterpret_cast<const char*>(&dataOffset), 4);    // data offset

    // DIB Header (BITMAPINFOHEADER)
    file.write(reinterpret_cast<const char*>(&dibHeaderSize), 4); // DIB header size
    file.write(reinterpret_cast<const char*>(&WIDTH), 4);         // width
    file.write(reinterpret_cast<const char*>(&HEIGHT), 4);        // height
    uint16_t planes = 1;
    file.write(reinterpret_cast<const char*>(&planes), 2);        // colour planes
    uint16_t bpp = 32;
    file.write(reinterpret_cast<const char*>(&bpp), 2);           // bits per pixel (RGBA)
    uint32_t compression = 0;
    file.write(reinterpret_cast<const char*>(&compression), 4);   // no compression
    file.write(reinterpret_cast<const char*>(&dataSize), 4);      // raw bitmap data size

    // Resolution & palette info (16 bytes total)
    uint8_t placeholder[16] = { 0 };
    file.write(reinterpret_cast<const char*>(placeholder), 16);

    for (int y = HEIGHT - 1; y >= 0; y--) {
        for (int x = 0; x < WIDTH; x++) {
            uint8_t r = static_cast<uint8_t>(imageBuffer[sycl::id<2>(x, y)].k * 255.0f);
            uint8_t g = static_cast<uint8_t>(imageBuffer[sycl::id<2>(x, y)].j * 255.0f);
            uint8_t b = static_cast<uint8_t>(imageBuffer[sycl::id<2>(x, y)].i * 255.0f);
            uint8_t a = 255;
            uint8_t pixel[4] = { r, g, b, a };
            file.write(reinterpret_cast<const char*>(pixel), 4);
        }
    }

    file.flush();
}

// The first kernal called
struct VertexShader {
    sycl::accessor<Vertex, 1, sycl::access::mode::read> verticesAcc;
    sycl::accessor<std::uint16_t, 1, sycl::access::mode::read> indexAcc;
    sycl::accessor<Matrix, 1, sycl::access::mode::read> transform;
    sycl::accessor<Tri, 1, sycl::access::mode::write> vertexOutAcc;

    void operator()(sycl::nd_item<1> triangle) const {
        size_t vertexIndex = indexAcc[triangle.get_global_id()];

        // apply the transform
        Vec4 position = transform[0] * Vec4(verticesAcc[vertexIndex].position, 1);
        position = position / position.w;
        //scale from screen space into pixels relative to the output size
        position.i = (position.i + 1) * (WIDTH / 2.f);
        position.j = (position.j + 1) * (HEIGHT / 2.f);

        // put the data in out buffer
        vertexOutAcc[triangle.get_group(0)].positions[triangle.get_local_id(0)] = { position.i, position.j, position.k };
        vertexOutAcc[triangle.get_group(0)].colours[triangle.get_local_id(0)] = verticesAcc[vertexIndex].colour;
    }
};

//The second kernal called
struct BasicPixelShader {
    sycl::accessor<Tri, 1, sycl::access::mode::read> vertexAcc;
    sycl::accessor<Vec3, 2, sycl::access::mode::write> imageAcc;

    void operator()(sycl::id<1> triangle) const {
        Vec3 p0 = vertexAcc[triangle.get(0)].positions[0];
        Vec3 p1 = vertexAcc[triangle.get(0)].positions[1];
        Vec3 p2 = vertexAcc[triangle.get(0)].positions[2];

        //Compute the bounding box of a triangle
        float minX = sycl::min(p0.i, sycl::min(p1.i, p2.i));
        float minY = sycl::min(p0.j, sycl::min(p1.j, p2.j));

        float maxX = sycl::max(p0.i, sycl::max(p1.i, p2.i));
        float maxY = sycl::max(p0.j, sycl::max(p1.j, p2.j));

        float denom = edge(p0, p1, p2);
        // makes sure the triangle is actually possible to render
        if (denom == 0)
            return;

        for (int y = static_cast<int>(sycl::floor(minY)); y < static_cast<int>(sycl::ceil(maxY)); y++) {
            if (y < 0 || y >= HEIGHT) continue;
            for (int x = static_cast<int>(sycl::floor(minX)); x < static_cast<int>(sycl::ceil(maxX)); x++) {
                if (x < 0 || x >= WIDTH) continue;
                // the last component is actually used but is need to make the function work.
                Vec3 pixel = { x + 0.5f , y + 0.5f, 0.f}; // sample the centre of the pixel

                float weight0 = edge(p1, p2, pixel) / denom;
                float weight1 = edge(p2, p0, pixel) / denom;
                float weight2 = edge(p0, p1, pixel) / denom;

                // checks if the point is in the triangle
                if (weight0 < 0.f || weight1 < 0 || weight2 < 0)
                    continue;

                // blends between the triangles
                Vec3 colour = vertexAcc[triangle.get(0)].colours[0] * weight0 + vertexAcc[triangle.get(0)].colours[1] * weight1 + vertexAcc[triangle.get(0)].colours[2] * weight2;

                imageAcc[sycl::id<2>(static_cast<size_t>(x), static_cast<size_t>(y))] = colour;
            }
        }
    }
};

// The improvement here is using an atmoic depth buffer for each pixel to mostly elementate a race condition 
// and ensure the nearest triangle is renderered
struct DepthBufferedPixelShader {
    sycl::accessor<Tri, 1, sycl::access::mode::read> vertexAcc;
    sycl::accessor<Vec3, 2, sycl::access::mode::write> imageAcc;
    sycl::accessor<float, 2, sycl::access::mode::read_write> depthAcc;


    void operator()(sycl::id<1> triangle) const {
        Vec3 p0 = vertexAcc[triangle.get(0)].positions[0];
        Vec3 p1 = vertexAcc[triangle.get(0)].positions[1];
        Vec3 p2 = vertexAcc[triangle.get(0)].positions[2];

        //Compute the bounding box of a triangle
        float minX = sycl::min(p0.i, sycl::min(p1.i, p2.i));
        float minY = sycl::min(p0.j, sycl::min(p1.j, p2.j));

        float maxX = sycl::max(p0.i, sycl::max(p1.i, p2.i));
        float maxY = sycl::max(p0.j, sycl::max(p1.j, p2.j));

        float denom = edge(p0, p1, p2);
        // makes sure the triangle is actually possible to render
        if (denom == 0)
            return;

        for (int y = static_cast<int>(sycl::floor(minY)); y < static_cast<int>(sycl::ceil(maxY)); y++) {
            if (y < 0 || y >= HEIGHT) continue;
            for (int x = static_cast<int>(sycl::floor(minX)); x < static_cast<int>(sycl::ceil(maxX)); x++) {
                if (x < 0 || x >= WIDTH) continue;
                // the last component is actually used but is need to make the function work.
                Vec3 pixel = { x + 0.5f , y + 0.5f, 0.f }; // sample the centre of the pixel

                float weight0 = edge(p1, p2, pixel) / denom;
                float weight1 = edge(p2, p0, pixel) / denom;
                float weight2 = edge(p0, p1, pixel) / denom;

                // checks if the point is in the triangle
                if (weight0 < 0.f || weight1 < 0 || weight2 < 0)
                    continue;

                // interpolate between the pixels to find the depth of the triangle at the given triangle
                float depth = p0.k * weight0 + p1.k * weight1 + p2.k * weight2;
                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> depthAtomic(depthAcc[sycl::id<2>(static_cast<size_t>(x), static_cast<size_t>(y))]);

                float expected = depthAtomic.load();
                bool won = false;
                do {
                    if (depth >= expected) break;
                } while (!(won = depthAtomic.compare_exchange_weak(expected, depth)));

                if (won) {
                    // blends between the triangles
                    Vec3 colour = vertexAcc[triangle.get(0)].colours[0] * weight0 + vertexAcc[triangle.get(0)].colours[1] * weight1 + vertexAcc[triangle.get(0)].colours[2] * weight2;
                    imageAcc[sycl::id<2>(static_cast<size_t>(x), static_cast<size_t>(y))] = colour;
                }
            }
        }
    }
};

void generateTriangles(const int& amount, std::vector<Vertex>& verticies, std::vector<std::uint16_t>& indicies) {
    int min = -30.0f;
    int max = 30.0f;

    std::mt19937 rng;

    std::uniform_real_distribution<float> positionDistribution(min, max);
    std::uniform_real_distribution<float> deltaInSize(-5.f, 5.f);
    std::uniform_real_distribution<float> colourDistribution(0.f, 1.f);

    for (int triangle = 0; triangle < amount; triangle++) {
        Vec3 p0(positionDistribution(rng), positionDistribution(rng), positionDistribution(rng));
        Vec3 p1(p0.i + deltaInSize(rng), p0.j + deltaInSize(rng), p0.k + deltaInSize(rng));
        Vec3 p2(p0.i + deltaInSize(rng), p0.j + deltaInSize(rng), p0.k + deltaInSize(rng));

        verticies.push_back({ p0, {colourDistribution(rng), colourDistribution(rng), colourDistribution(rng)} });
        verticies.push_back({ p1, {colourDistribution(rng), colourDistribution(rng), colourDistribution(rng)} });
        verticies.push_back({ p2, {colourDistribution(rng), colourDistribution(rng), colourDistribution(rng)} });

        indicies.push_back(verticies.size() - 3);
        indicies.push_back(verticies.size() - 2);
        indicies.push_back(verticies.size() - 1);
    }
}

void runTest(sycl::queue& queue, sycl::buffer<Vertex, 1>& vertexInBuffer, sycl::buffer<std::uint16_t>& indexBuffer, sycl::buffer<Tri, 1>& vertexOutBuffer, sycl::buffer<Matrix, 1>& transformBuffer, sycl::buffer<Vec3, 2>& imageBuffer, const int& indexCount) {

    std::cout << "Now Testing: " << indexCount / 3 << " triangles" << std::endl;
    std::uint64_t vertexDuration = 0;
    std::uint64_t pixelDuration = 0;

    // This is the vertex shader, all it does is apply the transform to the vertex which convert into screen coordinates
    // This may feel like an over kill way of doing this but its the most convient way of doing it "Nicely" without loosing information
    try {
        auto submitData = queue.submit([&](sycl::handler& handler) {
            auto verticesAcc = vertexInBuffer.get_access<sycl::access::mode::read>(handler);
            auto indexAcc = indexBuffer.get_access<sycl::access::mode::read>(handler);

            auto transform = transformBuffer.get_access<sycl::access::mode::read>(handler);
            auto vertexOutAcc = vertexOutBuffer.get_access<sycl::access::mode::write>(handler);

            // go triangle by triangle and perform the transform and move it into the triangle buffer
            handler.parallel_for(sycl::nd_range<1>(indexCount, 3), VertexShader{ verticesAcc , indexAcc, transform, vertexOutAcc });
        });
        queue.wait();

        auto vertexStart = submitData.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto vertexEnd = submitData.get_profiling_info<sycl::info::event_profiling::command_end>();
        vertexDuration = (vertexEnd - vertexStart) / 1'000'000;

        std::cout << "The vertex shader ran in " << vertexDuration << " milliseconds" << std::endl;
    }
    catch (sycl::exception e) {
        std::cerr << "[SYCL ERROR] " << e.what() << std::endl;
        __debugbreak();
    }

    try {
        //this is the pixel shader, it handles the colour of each triangle
        auto submitData = queue.submit([&](sycl::handler& handler) {
            auto vertexAcc = vertexOutBuffer.get_access<sycl::access::mode::read>(handler);

            auto imageAcc = imageBuffer.get_access<sycl::access::mode::write>(handler);

            size_t numberOfTriangles = indexCount / 3;
            //A work group for each triangle
            handler.parallel_for(sycl::range<1>(numberOfTriangles), BasicPixelShader{ vertexAcc, imageAcc });
        });
        queue.wait();

        auto pixelStart = submitData.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto pixelEnd = submitData.get_profiling_info<sycl::info::event_profiling::command_end>();
        pixelDuration = (pixelEnd - pixelStart) / 1'000'000;

        std::cout << "The pixel shader ran in " << pixelDuration << " milliseconds" << std::endl;

    }
    catch (sycl::exception e) {
        std::cerr << "[SYCL ERROR] " << e.what() << std::endl;
        __debugbreak();
    }

   
    std::cout << "Overall it ran in " << vertexDuration + pixelDuration << " milliseconds" << std::endl;
}

void runImprovedTest(sycl::queue& queue, sycl::buffer<Vertex, 1>& vertexInBuffer, sycl::buffer<std::uint16_t>& indexBuffer, sycl::buffer<Tri, 1>& vertexOutBuffer, sycl::buffer<Matrix, 1>& transformBuffer, sycl::buffer<Vec3, 2>& imageBuffer, sycl::buffer<float, 2>& depthBuffer, const int& indexCount) {
    std::cout << "Now Testing: " << indexCount / 3 << " triangles" << std::endl;
    std::uint64_t vertexDuration = 0;
    std::uint64_t pixelDuration = 0;

    // This is the vertex shader, all it does is apply the transform to the vertex which convert into screen coordinates
    // This may feel like an over kill way of doing this but its the most convient way of doing it "Nicely" without loosing information
    try {
        auto submitData = queue.submit([&](sycl::handler& handler) {
            auto verticesAcc = vertexInBuffer.get_access<sycl::access::mode::read>(handler);
            auto indexAcc = indexBuffer.get_access<sycl::access::mode::read>(handler);

            auto transform = transformBuffer.get_access<sycl::access::mode::read>(handler);
            auto vertexOutAcc = vertexOutBuffer.get_access<sycl::access::mode::write>(handler);

            // go triangle by triangle and perform the transform and move it into the triangle buffer
            handler.parallel_for(sycl::nd_range<1>(indexCount, 3), VertexShader{ verticesAcc , indexAcc, transform, vertexOutAcc });
            });
        queue.wait();

        auto vertexStart = submitData.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto vertexEnd = submitData.get_profiling_info<sycl::info::event_profiling::command_end>();
        vertexDuration = (vertexEnd - vertexStart) / 1'000'000;

        std::cout << "The vertex shader ran in " << vertexDuration << " milliseconds" << std::endl;
    }
    catch (sycl::exception e) {
        std::cerr << "[SYCL ERROR] " << e.what() << std::endl;
        __debugbreak();
    }

    try {
        //this is the pixel shader, it handles the colour of each triangle
        auto submitData = queue.submit([&](sycl::handler& handler) {
            auto vertexAcc = vertexOutBuffer.get_access<sycl::access::mode::read>(handler);
            auto imageAcc = imageBuffer.get_access<sycl::access::mode::write>(handler);
            auto depthAcc = depthBuffer.get_access<sycl::access::mode::read_write>(handler);

            size_t numberOfTriangles = indexCount / 3;
            //A work group for each triangle
            handler.parallel_for(sycl::range<1>(numberOfTriangles), DepthBufferedPixelShader{ vertexAcc, imageAcc, depthAcc });
            });
        queue.wait();

        auto pixelStart = submitData.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto pixelEnd = submitData.get_profiling_info<sycl::info::event_profiling::command_end>();
        pixelDuration = (pixelEnd - pixelStart) / 1'000'000;

        std::cout << "The improved pixel shader ran in " << pixelDuration << " milliseconds" << std::endl;

    }
    catch (sycl::exception e) {
        std::cerr << "[SYCL ERROR] " << e.what() << std::endl;
        __debugbreak();
    }


    std::cout << "Overall it ran in " << vertexDuration + pixelDuration << " milliseconds" << std::endl;
}

void performTests(bool improved, bool cpu=false){
    sycl::queue queue;
    if (cpu) {
        std::cout << "Testing the CPU" << std::endl;
        queue = sycl::queue(sycl::cpu_selector_v, sycl::property::queue::enable_profiling{});
    }
    else {
        queue = sycl::queue(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});
    }

    std::vector<Vertex> vertices;
    std::vector<std::uint16_t> indices;

    //This buffer doesnt change between tests
    Matrix finalTransform = GetViewMatrix({ 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0.f, 0.f, -20.f }) * GetPerspectiveMatrix(80.f, 0.1f, 1000.f);
    sycl::buffer<Matrix, 1> transformBuffer(&finalTransform, sycl::range<1>(1));

    // 100 triangle test
    {
        vertices.clear();
        indices.clear();
        generateTriangles(100, vertices, indices);

        sycl::buffer<Vertex, 1> vertexInBuffer(vertices.data(), sycl::range<1>(vertices.size()));
        sycl::buffer<std::uint16_t, 1> indexBuffer(indices.data(), sycl::range<1>(indices.size()));
        sycl::buffer<Tri, 1> vertexOutBuffer(sycl::range<1>(indices.size() / 3));
        sycl::buffer<Vec3, 2> imageBuffer(sycl::range<2>(WIDTH, HEIGHT));

        if (improved) {
            std::vector<float> depthData(WIDTH * HEIGHT, 1.f);
            sycl::buffer<float, 2> depthBuffer(depthData.data(), sycl::range<2>(WIDTH, HEIGHT));
            runImprovedTest(queue, vertexInBuffer, indexBuffer, vertexOutBuffer, transformBuffer, imageBuffer, depthBuffer, indices.size());
        }
        else {
            runTest(queue, vertexInBuffer, indexBuffer, vertexOutBuffer, transformBuffer, imageBuffer, indices.size());
        }


        sycl::host_accessor<Vec3, 2> imageAcc = imageBuffer.get_host_access();
        std::string fileName = std::to_string(100) + "triangles";
        if (cpu) {
            fileName += "CPU";
        }
        else {
            fileName += "GPU";
        }

        if (improved) {
            fileName += "Improved";
        }


        WriteImageToFile(fileName, imageAcc);
        std::cout << "Wrote the render to: " << fileName << ".bmp" << std::endl;
        std::cout << std::endl;
    }
    // 1000 triangle test
    {
        vertices.clear();
        indices.clear();
        generateTriangles(1000, vertices, indices);

        sycl::buffer<Vertex, 1> vertexInBuffer(vertices.data(), sycl::range<1>(vertices.size()));
        sycl::buffer<std::uint16_t, 1> indexBuffer(indices.data(), sycl::range<1>(indices.size()));
        sycl::buffer<Tri, 1> vertexOutBuffer(sycl::range<1>(indices.size() / 3));
        sycl::buffer<Vec3, 2> imageBuffer(sycl::range<2>(WIDTH, HEIGHT));

        if (improved) {
            std::vector<float> depthData(WIDTH * HEIGHT, 1.f);
            sycl::buffer<float, 2> depthBuffer(depthData.data(), sycl::range<2>(WIDTH, HEIGHT));
            runImprovedTest(queue, vertexInBuffer, indexBuffer, vertexOutBuffer, transformBuffer, imageBuffer, depthBuffer, indices.size());
        }
        else {
            runTest(queue, vertexInBuffer, indexBuffer, vertexOutBuffer, transformBuffer, imageBuffer, indices.size());
        }


        sycl::host_accessor<Vec3, 2> imageAcc = imageBuffer.get_host_access();
        std::string fileName = std::to_string(1000) + "triangles";
        
        if (cpu) {
            fileName += "CPU";
        }
        else {
            fileName += "GPU";
        }

        if (improved) {
            fileName += "Improved";
        }

        WriteImageToFile(fileName, imageAcc);
        std::cout << "Wrote the render to: " << fileName << ".bmp" << std::endl;
        std::cout << std::endl;
    }
    // 10,0000 triangle test
    {
        vertices.clear();
        indices.clear();
        generateTriangles(10000, vertices, indices);

        sycl::buffer<Vertex, 1> vertexInBuffer(vertices.data(), sycl::range<1>(vertices.size()));
        sycl::buffer<std::uint16_t, 1> indexBuffer(indices.data(), sycl::range<1>(indices.size()));
        sycl::buffer<Tri, 1> vertexOutBuffer(sycl::range<1>(indices.size() / 3));
        sycl::buffer<Vec3, 2> imageBuffer(sycl::range<2>(WIDTH, HEIGHT));

        if (improved) {
            std::vector<float> depthData(WIDTH * HEIGHT, 1.f);
            sycl::buffer<float, 2> depthBuffer(depthData.data(), sycl::range<2>(WIDTH, HEIGHT));
            runImprovedTest(queue, vertexInBuffer, indexBuffer, vertexOutBuffer, transformBuffer, imageBuffer, depthBuffer, indices.size());
        }
        else {
            runTest(queue, vertexInBuffer, indexBuffer, vertexOutBuffer, transformBuffer, imageBuffer, indices.size());
        }


        sycl::host_accessor<Vec3, 2> imageAcc = imageBuffer.get_host_access();
        std::string fileName = std::to_string(10000) + "triangles";
        
        if (cpu) {
            fileName += "CPU";
        }
        else {
            fileName += "GPU";
        }

        if (improved) {
            fileName += "Improved";
        }

        WriteImageToFile(fileName, imageAcc);       
        std::cout << "Wrote the render to: " << fileName << ".bmp" << std::endl;
        std::cout << std::endl;
    }
}  

int main() {
    performTests(false, true);
    
    std::cout << std::endl << "Now conduction the improved tests" << std::endl;
    performTests(true, true);

    return 0;
}
