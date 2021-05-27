#pragma once

#include <string>

enum StreamFormat {
    STREAM_FORMAT_RAW = 0,
    STREAM_FORMAT_OCTOWS2811 = 1,
};

class NetworkStream {
public:
    NetworkStream();
    ~NetworkStream();

    void start(std::string &host, std::string &port, StreamFormat format);
    void stop();
    bool isStreaming();

    int sendFrame(unsigned char *pixels, int width, int height, int channels);

private:
    std::string  m_host;
    std::string  m_port;
    int          m_socket;
    StreamFormat m_format;

    bool         m_streaming;

    int sendOctoWS2811Frame(unsigned char *pixels, int width, int height, int channels);
    int sendRawFrame(unsigned char *data, int data_len);

    //TODO: send thread
};
