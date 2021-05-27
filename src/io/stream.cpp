//TODO: Test Windows,Linux,Mac

#ifdef _WIN32
  #include <winsock2.h>
  #include <Ws2tcpip.h>
#else
  /* Assume that any non-Windows platform uses POSIX-style sockets instead. */
  #include <sys/socket.h>
  #include <arpa/inet.h>
  #include <netdb.h>  /* Needed for getaddrinfo() and freeaddrinfo() */
  #include <unistd.h> /* Needed for close() */
#endif


#include <iostream>
#include <cstring>
#include <memory>

#include "stream.h"

NetworkStream::NetworkStream() {
}

NetworkStream::~NetworkStream() {
    stop();

}

void closeTcpStream(int sockfd) {
#ifdef _WIN32
    closesocket(new_conn_fd);
#else
    close(sockfd);
#endif

#ifdef _WIN32
    WSACleanup();
#endif
}

int connectTcpStream(const char *host, const char *port) {
    int sockfd;
    struct addrinfo hints, *servinfo, *p;
    int rv;

#ifdef _WIN32
    WSADATA wsa_data;
    int rv = WSAStartup(MAKEWORD(2,2), &wsa_data);
    if (rv != NO_ERROR) {
        std::cout << "WSAStartup failed with error: " << rv << std::endl;
        return -1;
    }
#endif

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if ((rv = getaddrinfo(host, port, &hints, &servinfo)) != 0) {
        std::cout << "getaddrinfo: " << gai_strerror(rv) << std::endl;
        return -1;
    }

    // Connect to the first we can address we can
    for(p = servinfo; p != NULL; p = p->ai_next) {
        if ((sockfd = socket(p->ai_family, p->ai_socktype,
                p->ai_protocol)) == -1) {
            std::cout << "client: socket " << strerror(errno) << std::endl;
            continue;
        }

        if (connect(sockfd, p->ai_addr, p->ai_addrlen) == -1) {
            closeTcpStream(sockfd);
            std::cout << "client: connect " << strerror(errno) << std::endl;
            continue;
        }

        break;
    }

    if (p == NULL) {
        std::cout << "client: failed to connect" << std::endl;
        return -1;
    }

    freeaddrinfo(servinfo);

    return sockfd;
}

int sendall(int sockfd, unsigned char *buf, int len)
{
    int totalSent = 0;
    int bytesToSend = len;
    int n;

    while(totalSent < len) {
        n = send(sockfd, buf+totalSent, bytesToSend, MSG_NOSIGNAL);
        if (n < 0) {
            return -1;
        }
        totalSent += n;
        bytesToSend -= n;
    }

    return totalSent;
}

//TODO: Use htonl
int buffer_write_long(unsigned char *buf, uint32_t d) {
    *(uint32_t*)buf = htonl(d);
    // buf[0] = (d>>24)&0xff;
    // buf[1] = (d>>16)&0xff;
    // buf[2] = (d>>8)&0xff;
    // buf[3] = d&0xff;
    return 4;
}

void NetworkStream::start(std::string &host, std::string &port, StreamFormat format) {
    if (m_streaming) {
        std::cout << "Found existing stream." << std::endl;
        stop();
    }

    // Connect to socket
    m_socket = connectTcpStream(host.c_str(), port.c_str());
    if (m_socket < 0) {
        m_socket = -1;
        std::cout << "Failed to connect to " << host << ":" << port << std::endl;
        return;
    }

    std::cout << "Started stream to " << host << ":" << port << std::endl;

    m_streaming = true;
    m_host = host;
    m_port = port;
    m_format = format;
}

void NetworkStream::stop() {
    // Close socket
    if (m_streaming) {
        std::cout << "Stopping stream to " << m_host << ":" << m_port << std::endl;
        closeTcpStream(m_socket);
    }

    m_streaming = false;
    m_host = "";
    m_port = "";
    m_socket = -1;
}

bool NetworkStream::isStreaming() {
	return m_streaming;
}

int NetworkStream::sendFrame(unsigned char *pixels, int width, int height, int channels) {
	switch(m_format) {
		case STREAM_FORMAT_RAW:
			return sendRawFrame(pixels, width * height * channels);
		case STREAM_FORMAT_OCTOWS2811:
			return sendOctoWS2811Frame(pixels, width, height, channels);
	}

	std::cout << "Unsupported stream format?!" << std::endl;
	stop();
	return -1;
}

int NetworkStream::sendRawFrame(unsigned char *data, int data_len) {
    const int header_len = 4;
    unsigned char header[header_len];

    buffer_write_long(header, data_len);

    // Send frame header
    int sent = sendall(m_socket, header, header_len);
    if (sent < 0) {
        std::cout << "Failed to stream frame header" << std::endl;
        stop();
        return -1;
    }

    // Send frame data
    sent = sendall(m_socket, data, data_len);
    if (sent < 0) {
        std::cout << "Failed to stream frame data" << std::endl;
        stop();
        return -1;
    }

    return 0;
}

// Sends data encoded in a format that the OctoWS2811 LED library supports.
int NetworkStream::sendOctoWS2811Frame(unsigned char *pixels, int width, int height, int channels) {
    //NOTE: Assumes 8-bit channels and the first 3 channels are RGB.
    if (channels < 3) {
        std::cout << "Streaming OctoWS2811 formated data requires at least 3 color channels (RGB)" << std::endl;
        stop();
        return -1;
    }

    //NOTE: OctoWS2811 actually supports multiples of 8 rows, but it's a slightly more
    //      complicated encoding that this code does not currently do.
    if (height > 8) {
        std::cout << "Streaming OctoWS2811 formated data supports a maximum of 8 rows" << std::endl;
        stop();
        return -1;
    }

    // Width pixels * 8 rows * 3 channels
    int frame_len = width * 8 * 3;

    // 4 header bytes + frame data
    const int header_len = 4;
    int buf_len = header_len + frame_len;

    auto buf = std::unique_ptr<unsigned char[]>(new unsigned char [buf_len]);
    memset(buf.get(), 0, buf_len);

    unsigned char *header = buf.get();
    unsigned char *frame = buf.get() + header_len;

    // Pack frame header
    buffer_write_long(header, frame_len);

    // Pack frame data
    //
    // The format is a bit odd.
    //
    // All color data for row 0 is encoded in bit 0 of every output byte.
    // All color data for row 1 is encoded in bit 1 of every output byte.
    // <...>
    // All color data for row 7 is encoded in bit 7 of every output byte.
    //
    // A single 8-bit color channel of input is spread out and encoded over
    // 8 bytes. One bit per byte. It takes 24 bytes to encode 3 color channels.
    //
    // To comlicate matters more, color data is expected in GRB order.
    //
    // So, the first 24 bytes of output encodes the first pixel of every row.
    // The second 24 bytes encodes the second pixel of every row. Etc.
    //
    // I'll assume you understand now and move on :-)
    const int mapInputToOutputChannel[3] = {1, 0, 2};

    // Loop over input and compute output.
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {

            // Index into the input buffer of the current pixel's color data
            int i = y * width + x;
            i *= channels;

            for (int channel = 0; channel < 3; channel++) {
                int output_channel = mapInputToOutputChannel[channel];

                for (int bit = 0; bit < 8; bit++) {
                    int channel_byte = pixels[i+channel];
                    int channel_bit = 0x1 & (channel_byte >> bit);

                    // Which byte in the output buffer
                    int output_byte = x * 8 * 3 + (7 - bit) + (8 * output_channel);
                    // Which bit in the output byte
                    int output_bit = y;

                    // Combine with output buffer
                    frame[output_byte] |= channel_bit << output_bit;
                }

            }

        }
    }

    // Send frame
    std::cout << "Sending " << buf_len << " bytes" << std::endl;
    int sent = sendall(m_socket, buf.get(), buf_len);
    if (sent < 0) {
        std::cout << "Failed to stream frame" << std::endl;
        stop();
        return -1;
    }

    return 0;
}
