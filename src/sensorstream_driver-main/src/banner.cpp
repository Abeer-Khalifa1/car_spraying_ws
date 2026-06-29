#include "sensorstream_driver/banner.hpp"

#include <cstdio>
#include <string>

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>

#include "qrcodegen.hpp"

namespace {

static void print_qr(const qrcodegen::QrCode& qr) {
  int sz = qr.getSize();
  // Print with 4-module quiet zone. Two rows of modules packed into one terminal
  // row using half-block characters (▀ ▄ █ space).
  // Convention: dark module = filled, light module = empty.
  // On a dark terminal: dark=space, light=block character.
  printf("\n");
  // Top quiet zone (2 terminal rows = 4 modules)
  for (int q = 0; q < 2; ++q) {
    printf("    ");
    for (int col = -4; col < sz + 4; ++col) printf("█");
    printf("\n");
  }
  for (int row = 0; row < sz; row += 2) {
    printf("    ");
    // Left quiet zone
    printf("████");
    for (int col = 0; col < sz; ++col) {
      bool top = qr.getModule(col, row);
      bool bot = (row + 1 < sz) ? qr.getModule(col, row + 1) : false;
      if      (!top && !bot) printf("█");
      else if (!top &&  bot) printf("▀");
      else if ( top && !bot) printf("▄");
      else                   printf(" ");
    }
    // Right quiet zone
    printf("████");
    printf("\n");
  }
  // Bottom quiet zone
  for (int q = 0; q < 2; ++q) {
    printf("    ");
    for (int col = -4; col < sz + 4; ++col) printf("█");
    printf("\n");
  }
  printf("\n");
}

static std::string get_local_ip() {
  struct ifaddrs* ifaddr = nullptr;
  if (getifaddrs(&ifaddr) != 0) return "127.0.0.1";

  std::string result = "127.0.0.1";
  for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET) continue;
    std::string name = ifa->ifa_name ? ifa->ifa_name : "";
    if (name == "lo" || name == "lo0") continue;
    char buf[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &reinterpret_cast<sockaddr_in*>(ifa->ifa_addr)->sin_addr,
              buf, sizeof(buf));
    result = buf;
    if (name.rfind("en", 0) == 0 || name.rfind("wl", 0) == 0) break;
  }
  freeifaddrs(ifaddr);
  return result;
}

}  // anonymous namespace

namespace sensorstream {

void print_banner(int wifi_port, int usb_port) {
  const char* BOLD  = "\033[1m";
  const char* CYAN  = "\033[36m";
  const char* GREEN = "\033[32m";
  const char* DIM   = "\033[2m";
  const char* RESET = "\033[0m";

  std::string ip        = get_local_ip();
  std::string wifi_addr = ip + ":" + std::to_string(wifi_port);

  printf("\n");
  printf("  %s%s╔══════════════════════════════════════╗%s\n", BOLD, CYAN, RESET);
  printf("  %s%s║          SensorStream Driver         ║%s\n", BOLD, CYAN, RESET);
  printf("  %s%s╚══════════════════════════════════════╝%s\n", BOLD, CYAN, RESET);
  printf("\n");

  printf("  %s%sWiFi%s  %s%s%s%s\n",   BOLD, GREEN, RESET, BOLD, CYAN, wifi_addr.c_str(), RESET);
  printf("  %s%sUSB port %s%s%s %d\n",
         BOLD, GREEN, RESET, BOLD, CYAN, usb_port);
  printf("\n");

  printf("  %sScan to connect (WiFi):%s\n", DIM, RESET);
  try {
    auto qr = qrcodegen::QrCode::encodeText(wifi_addr.c_str(), qrcodegen::QrCode::Ecc::MEDIUM);
    print_qr(qr);
  } catch (const std::exception& e) {
    fprintf(stderr, "  [sensorstream] QR generation failed: %s\n", e.what());
    std::string border(wifi_addr.size() + 4, '-');
    printf("    +%s+\n", border.c_str());
    printf("    |  %s  |\n", wifi_addr.c_str());
    printf("    +%s+\n\n", border.c_str());
  }

  fflush(stdout);
}

}  // namespace sensorstream
