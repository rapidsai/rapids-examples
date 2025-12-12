/**
 * UCXX cuDF Example
 *
 * This example demonstrates how to:
 * 1. Create a simple cuDF integer column using cudf::sequence
 * 2. Send the column data directly from GPU memory using UCXX
 * 3. Receive the data into GPU memory
 * 4. Verify the received data matches (copying to host only for verification)
 *
 * Similar to ucxx/cpp/examples/basic.cpp but transfers cuDF column data
 * directly between GPU buffers.
 */

#include <cassert>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unistd.h>
#include <vector>

// UCXX headers
#include <ucxx/api.h>
#include <ucxx/buffer.h>
#include <ucxx/utils/sockaddr.h>
#include <ucxx/utils/ucx.h>

// cuDF headers
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

// RMM headers
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

// =============================================================================
// Column Utilities
// =============================================================================

/**
 * @brief Copy column data from device to host (only used for verification/printing)
 */
std::vector<int32_t> columnToHost(cudf::column_view const& column)
{
  std::vector<int32_t> host_buffer(column.size());
  cudaMemcpy(host_buffer.data(),
             column.data<int32_t>(),
             column.size() * sizeof(int32_t),
             cudaMemcpyDeviceToHost);
  cudaStreamSynchronize(0);
  return host_buffer;
}

/**
 * @brief Create a cuDF column from a device buffer
 *
 * @param device_buffer  RMM device buffer containing the data (ownership transferred)
 * @param num_elements   Number of int32 elements in the buffer
 */
std::unique_ptr<cudf::column> deviceBufferToColumn(rmm::device_buffer&& device_buffer,
                                                   cudf::size_type num_elements)
{
  return std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::INT32),
                                        num_elements,
                                        std::move(device_buffer),
                                        rmm::device_buffer{},  // no null mask
                                        0);                    // null count
}

/**
 * @brief Allocate a device buffer for receiving column data
 */
rmm::device_buffer allocateDeviceBuffer(size_t size_bytes)
{
  return rmm::device_buffer(size_bytes, cudf::get_default_stream());
}

/**
 * @brief Create a cuDF integer sequence column [0, 1, 2, ..., size-1]
 */
std::unique_ptr<cudf::column> createSequenceColumn(cudf::size_type size)
{
  cudf::numeric_scalar<int32_t> init_scalar(0, true, cudf::get_default_stream());
  cudf::numeric_scalar<int32_t> step_scalar(1, true, cudf::get_default_stream());
  return cudf::sequence(size, init_scalar, step_scalar);
}

/**
 * @brief Print preview of column data (first/last few elements)
 */
void printColumnPreview(std::vector<int32_t> const& data, std::string const& label = "")
{
  if (!label.empty()) std::cout << label << " ";

  const int preview_count = std::min(5, static_cast<int>(data.size()));
  std::cout << "[";

  for (int i = 0; i < preview_count; ++i) {
    std::cout << data[i];
    if (i < preview_count - 1) std::cout << ", ";
  }

  if (static_cast<int>(data.size()) > preview_count * 2) {
    std::cout << ", ...";
  }

  if (static_cast<int>(data.size()) > preview_count) {
    std::cout << ", ";
    int start = std::max(preview_count, static_cast<int>(data.size()) - preview_count);
    for (size_t i = start; i < data.size(); ++i) {
      std::cout << data[i];
      if (i < data.size() - 1) std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

/**
 * @brief Verify two columns have identical data (copies to host for comparison)
 */
bool verifyColumnsMatch(cudf::column_view const& expected, cudf::column_view const& actual)
{
  if (expected.size() != actual.size()) {
    std::cerr << "Size mismatch: expected " << expected.size() << ", got " << actual.size()
              << std::endl;
    return false;
  }

  // Copy both to host for verification
  auto expected_host = columnToHost(expected);
  auto actual_host   = columnToHost(actual);

  for (cudf::size_type i = 0; i < expected.size(); ++i) {
    if (expected_host[i] != actual_host[i]) {
      std::cerr << "Data mismatch at index " << i << ": expected " << expected_host[i] << ", got "
                << actual_host[i] << std::endl;
      return false;
    }
  }
  return true;
}

// =============================================================================
// UCXX Communication Layer
// =============================================================================

/**
 * @brief Manages a UCXX connection between a listener and client endpoint
 *
 * This class encapsulates the setup and teardown of UCXX communication,
 * providing simple send/receive methods for transferring data.
 * Supports both host and device memory transfers.
 */
class UCXXConnection {
 public:
  /**
   * @brief Construct a new UCXX connection on the specified port
   */
  explicit UCXXConnection(uint16_t port) : _port(port)
  {
    // Create UCXX context and worker
    _context = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
    _worker  = _context->createWorker();

    // Setup listener
    _listener = _worker->createListener(_port, listenerCallback, this);

    // Create client endpoint that connects to our own listener (loopback)
    _client_endpoint = _worker->createEndpointFromHostname("127.0.0.1", _port, true);

    // Wait for connection to be established
    while (!_server_endpoint) {
      _worker->progress();
    }

    // Perform wireup exchange (using host memory for small control message)
    performWireup();
  }

  ~UCXXConnection() = default;

  // Disable copy
  UCXXConnection(UCXXConnection const&)            = delete;
  UCXXConnection& operator=(UCXXConnection const&) = delete;

  /**
   * @brief Send data from server to client (supports device memory)
   */
  void sendToClient(void const* data, size_t size, uint64_t tag)
  {
    // Ensure any pending GPU work is complete before sending
    cudaStreamSynchronize(0);
    auto request = _server_endpoint->tagSend(const_cast<void*>(data), size, ucxx::Tag{tag});
    waitForRequest(request);
  }

  /**
   * @brief Receive data on client from server (supports device memory)
   */
  void recvOnClient(void* buffer, size_t size, uint64_t tag)
  {
    auto request = _client_endpoint->tagRecv(buffer, size, ucxx::Tag{tag}, ucxx::TagMaskFull);
    waitForRequest(request);
    // Ensure receive is complete before GPU uses the data
    cudaStreamSynchronize(0);
  }

  /**
   * @brief Send data from client to server (supports device memory)
   */
  void sendToServer(void const* data, size_t size, uint64_t tag)
  {
    cudaStreamSynchronize(0);
    auto request = _client_endpoint->tagSend(const_cast<void*>(data), size, ucxx::Tag{tag});
    waitForRequest(request);
  }

  /**
   * @brief Receive data on server from client (supports device memory)
   */
  void recvOnServer(void* buffer, size_t size, uint64_t tag)
  {
    auto request = _server_endpoint->tagRecv(buffer, size, ucxx::Tag{tag}, ucxx::TagMaskFull);
    waitForRequest(request);
    cudaStreamSynchronize(0);
  }

  /**
   * @brief Get the port this connection is using
   */
  uint16_t port() const { return _port; }

 private:
  uint16_t _port;
  std::shared_ptr<ucxx::Context> _context;
  std::shared_ptr<ucxx::Worker> _worker;
  std::shared_ptr<ucxx::Listener> _listener;
  std::shared_ptr<ucxx::Endpoint> _server_endpoint;  // Created from connection request
  std::shared_ptr<ucxx::Endpoint> _client_endpoint;  // Created by connecting to listener

  void waitForRequest(std::shared_ptr<ucxx::Request> const& request)
  {
    while (!request->isCompleted()) {
      _worker->progress();
    }
    request->checkError();
  }

  void performWireup()
  {
    // Small exchange to let UCX identify capabilities (host memory for control)
    int32_t wireup_send = 42;
    int32_t wireup_recv = 0;
    auto send_req = _server_endpoint->tagSend(&wireup_send, sizeof(int32_t), ucxx::Tag{0});
    auto recv_req =
      _client_endpoint->tagRecv(&wireup_recv, sizeof(int32_t), ucxx::Tag{0}, ucxx::TagMaskFull);
    waitForRequest(send_req);
    waitForRequest(recv_req);
  }

  static void listenerCallback(ucp_conn_request_h conn_request, void* arg)
  {
    auto* self = reinterpret_cast<UCXXConnection*>(arg);

    // Log connection
    char ip_str[INET6_ADDRSTRLEN];
    char port_str[INET6_ADDRSTRLEN];
    ucp_conn_request_attr_t attr{};
    attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
    ucp_conn_request_query(conn_request, &attr);
    ucxx::utils::sockaddr_get_ip_port_str(&attr.client_address, ip_str, port_str, INET6_ADDRSTRLEN);
    std::cout << "Connection established from " << ip_str << ":" << port_str << std::endl;

    // Create server endpoint from connection request
    self->_server_endpoint = self->_listener->createEndpointFromConnRequest(conn_request, true);
  }
};

// =============================================================================
// Column Transfer API (Device-to-Device)
// =============================================================================

/**
 * @brief Transfer a cuDF column over UCXX using device memory directly
 *
 * Sends data directly from the source column's device buffer and receives
 * into a newly allocated device buffer. No host copies during transfer.
 *
 * @param conn       The UCXX connection to use
 * @param column     The column to transfer (data stays on GPU)
 * @param tag        Message tag for this transfer
 * @return           New column created from received device data
 */
std::unique_ptr<cudf::column> transferColumn(UCXXConnection& conn,
                                             cudf::column_view const& column,
                                             uint64_t tag = 1)
{
  size_t data_size    = column.size() * sizeof(int32_t);
  auto num_elements   = column.size();

  // Get pointer to source data on device
  void const* send_ptr = column.data<int32_t>();

  // Allocate receive buffer on device
  auto recv_buffer = allocateDeviceBuffer(data_size);
  void* recv_ptr   = recv_buffer.data();

  // Transfer directly between device buffers: server sends, client receives
  conn.sendToClient(send_ptr, data_size, tag);
  conn.recvOnClient(recv_ptr, data_size, tag);

  // Create new column from received device buffer
  return deviceBufferToColumn(std::move(recv_buffer), num_elements);
}

/**
 * @brief Transfer a cuDF column and verify it matches the original
 *
 * Transfers data directly between GPU buffers using UCXX.
 * Only copies to host for verification/printing.
 *
 * @param conn       The UCXX connection to use
 * @param column     The column to transfer and verify
 * @param tag        Message tag for this transfer
 * @param verbose    Print progress messages
 * @return           true if transfer succeeded and data matches
 */
bool transferAndVerify(UCXXConnection& conn,
                       cudf::column_view const& column,
                       uint64_t tag = 1,
                       bool verbose = true)
{
  size_t data_size = column.size() * sizeof(int32_t);

  if (verbose) {
    std::cout << "Transferring column: " << column.size() << " elements, " << data_size
              << " bytes" << std::endl;
    std::cout << "  Transfer mode: GPU-to-GPU (device buffers)" << std::endl;

    // Copy to host only for display
    std::cout << "  Send data: ";
    printColumnPreview(columnToHost(column));
  }

  // Transfer the column (device-to-device)
  auto received = transferColumn(conn, column, tag);

  if (verbose) {
    // Copy to host only for display
    std::cout << "  Recv data: ";
    printColumnPreview(columnToHost(received->view()));
  }

  // Verify by copying both to host and comparing
  bool success = verifyColumnsMatch(column, received->view());

  if (verbose) {
    if (success) {
      std::cout << "  Verification: PASSED (" << column.size() << " elements match)" << std::endl;
    } else {
      std::cout << "  Verification: FAILED" << std::endl;
    }
  }

  return success;
}

// =============================================================================
// Command Line Parsing
// =============================================================================

struct Args {
  uint16_t port{12345};
  int32_t size{1000};
  bool help{false};

  bool parse(int argc, char* const argv[])
  {
    int c;
    while ((c = getopt(argc, argv, "p:s:h")) != -1) {
      switch (c) {
        case 'p':
          port = static_cast<uint16_t>(atoi(optarg));
          break;
        case 's':
          size = atoi(optarg);
          break;
        case 'h':
        default:
          help = true;
          return false;
      }
    }
    return port > 0 && size > 0;
  }

  static void printUsage()
  {
    std::cerr << "Usage: ucxx_cudf_example [options]\n"
              << "\n"
              << "Transfers a cuDF integer column via UCXX using GPU memory directly.\n"
              << "Data is sent/received in device buffers; only copied to host for verification.\n"
              << "\n"
              << "Options:\n"
              << "  -p <port>   Port number (default: 12345)\n"
              << "  -s <size>   Number of elements (default: 1000)\n"
              << "  -h          Show this help\n";
  }
};

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv)
{
  // Parse arguments
  Args args;
  if (!args.parse(argc, argv)) {
    Args::printUsage();
    return args.help ? 0 : 1;
  }

  std::cout << "=== UCXX cuDF Example (GPU-to-GPU Transfer) ===" << std::endl;
  std::cout << "Port: " << args.port << ", Size: " << args.size << std::endl;
  std::cout << std::endl;

  // Create test column on GPU
  std::cout << "Creating cuDF sequence column [0, 1, 2, ..., " << (args.size - 1) << "] on GPU"
            << std::endl;
  auto column = createSequenceColumn(args.size);

  // Setup UCXX connection
  std::cout << std::endl << "Setting up UCXX connection..." << std::endl;
  UCXXConnection conn(args.port);

  // Transfer and verify (GPU-to-GPU, verify on host)
  std::cout << std::endl;
  bool success = transferAndVerify(conn, column->view());

  std::cout << std::endl;
  if (success) {
    std::cout << "=== Example completed successfully ===" << std::endl;
    return 0;
  } else {
    std::cout << "=== Example FAILED ===" << std::endl;
    return 1;
  }
}

