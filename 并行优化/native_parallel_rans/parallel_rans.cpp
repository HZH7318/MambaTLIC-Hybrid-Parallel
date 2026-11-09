#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "rans64.h"

namespace py = pybind11;

namespace {
constexpr int kPrecision = 16;
constexpr uint16_t kBypassPrecision = 4;
constexpr uint16_t kMaxBypass = (1 << kBypassPrecision) - 1;

struct EncSymbol {
  uint16_t start;
  uint16_t range;
  bool bypass;
};

struct TaskView {
  const int32_t *symbols;
  const int32_t *indexes;
  ssize_t size;
};

struct IndexView {
  const int32_t *indexes;
  ssize_t size;
};

struct CdfView {
  const int32_t *data;
  const int32_t *lengths;
  const int32_t *offsets;
  ssize_t rows;
  ssize_t stride;
};

inline void put_bits(Rans64State *r, uint32_t **ptr, uint32_t value,
                     uint32_t bits) {
  uint64_t x = *r;
  const uint32_t freq = 1 << (16 - bits);
  const uint64_t x_max = ((RANS64_L >> 16) << 32) * freq;
  if (x >= x_max) {
    *ptr -= 1;
    **ptr = static_cast<uint32_t>(x);
    x >>= 32;
  }
  *r = (x << bits) | value;
}

inline uint32_t get_bits(Rans64State *r, uint32_t **ptr, uint32_t bits) {
  uint64_t x = *r;
  const uint32_t value = x & ((1u << bits) - 1);
  x >>= bits;
  if (x < RANS64_L) {
    x = (x << 32) | **ptr;
    *ptr += 1;
  }
  *r = x;
  return value;
}

void validate_index(int32_t index, const CdfView &cdf) {
  if (index < 0 || index >= cdf.rows) {
    throw std::runtime_error("CDF index is out of range");
  }
}

std::string encode_one(const TaskView &task, const CdfView &cdf) {
  std::vector<EncSymbol> encoded;
  encoded.reserve(static_cast<size_t>(task.size) * 2);
  for (ssize_t i = 0; i < task.size; ++i) {
    const int32_t cdf_index = task.indexes[i];
    validate_index(cdf_index, cdf);
    const int32_t cdf_length = cdf.lengths[cdf_index];
    const int32_t max_value = cdf_length - 2;
    int32_t value = task.symbols[i] - cdf.offsets[cdf_index];
    uint32_t raw_value = 0;
    if (value < 0) {
      raw_value = static_cast<uint32_t>(-2 * value - 1);
      value = max_value;
    } else if (value >= max_value) {
      raw_value = static_cast<uint32_t>(2 * (value - max_value));
      value = max_value;
    }

    const int32_t *row = cdf.data + cdf_index * cdf.stride;
    const int32_t start = row[value];
    const int32_t range = row[value + 1] - start;
    if (range <= 0) {
      throw std::runtime_error("CDF must be strictly increasing");
    }
    encoded.push_back({static_cast<uint16_t>(start),
                       static_cast<uint16_t>(range), false});

    if (value == max_value) {
      int32_t bypasses = 0;
      while ((raw_value >> (bypasses * kBypassPrecision)) != 0) {
        ++bypasses;
      }
      int32_t remaining = bypasses;
      while (remaining >= kMaxBypass) {
        encoded.push_back({kMaxBypass, static_cast<uint16_t>(kMaxBypass + 1),
                           true});
        remaining -= kMaxBypass;
      }
      encoded.push_back({static_cast<uint16_t>(remaining),
                         static_cast<uint16_t>(remaining + 1), true});
      for (int32_t j = 0; j < bypasses; ++j) {
        const int32_t part =
            (raw_value >> (j * kBypassPrecision)) & kMaxBypass;
        encoded.push_back({static_cast<uint16_t>(part),
                           static_cast<uint16_t>(part + 1), true});
      }
    }
  }

  Rans64State rans;
  Rans64EncInit(&rans);
  std::vector<uint32_t> output(std::max<size_t>(encoded.size() + 2, 4),
                               0xCCCCCCCC);
  uint32_t *ptr = output.data() + output.size();
  while (!encoded.empty()) {
    const EncSymbol symbol = encoded.back();
    encoded.pop_back();
    if (symbol.bypass) {
      put_bits(&rans, &ptr, symbol.start, kBypassPrecision);
    } else {
      Rans64EncPut(&rans, &ptr, symbol.start, symbol.range, kPrecision);
    }
  }
  Rans64EncFlush(&rans, &ptr);
  const size_t bytes =
      static_cast<size_t>(output.data() + output.size() - ptr) * sizeof(uint32_t);
  return std::string(reinterpret_cast<const char *>(ptr), bytes);
}

std::vector<int32_t> decode_one(const std::string &stream,
                                const IndexView &task, const CdfView &cdf) {
  if (stream.size() < sizeof(uint64_t)) {
    throw std::runtime_error("Invalid rANS stream");
  }
  std::string owned = stream;
  uint32_t *ptr = reinterpret_cast<uint32_t *>(owned.data());
  Rans64State rans;
  Rans64DecInit(&rans, &ptr);
  std::vector<int32_t> output(static_cast<size_t>(task.size));

  for (ssize_t i = 0; i < task.size; ++i) {
    const int32_t cdf_index = task.indexes[i];
    validate_index(cdf_index, cdf);
    const int32_t length = cdf.lengths[cdf_index];
    const int32_t max_value = length - 2;
    const int32_t *row = cdf.data + cdf_index * cdf.stride;
    const uint32_t cumulative = Rans64DecGet(&rans, kPrecision);
    const int32_t *upper = std::upper_bound(row, row + length, cumulative);
    if (upper == row || upper == row + length) {
      throw std::runtime_error("Malformed CDF or rANS stream");
    }
    int32_t value = static_cast<int32_t>(upper - row - 1);
    Rans64DecAdvance(&rans, &ptr, row[value], row[value + 1] - row[value],
                     kPrecision);

    if (value == max_value) {
      int32_t part = get_bits(&rans, &ptr, kBypassPrecision);
      int32_t bypasses = part;
      while (part == kMaxBypass) {
        part = get_bits(&rans, &ptr, kBypassPrecision);
        bypasses += part;
      }
      int32_t raw_value = 0;
      for (int32_t j = 0; j < bypasses; ++j) {
        part = get_bits(&rans, &ptr, kBypassPrecision);
        raw_value |= part << (j * kBypassPrecision);
      }
      value = raw_value >> 1;
      value = (raw_value & 1) ? -value - 1 : value + max_value;
    }
    output[static_cast<size_t>(i)] = value + cdf.offsets[cdf_index];
  }
  return output;
}

CdfView make_cdf_view(const py::array_t<int32_t, py::array::c_style> &cdf,
                      const py::array_t<int32_t, py::array::c_style> &lengths,
                      const py::array_t<int32_t, py::array::c_style> &offsets) {
  if (cdf.ndim() != 2 || lengths.ndim() != 1 || offsets.ndim() != 1 ||
      lengths.shape(0) != cdf.shape(0) || offsets.shape(0) != cdf.shape(0)) {
    throw std::invalid_argument("Invalid CDF table shapes");
  }
  return {cdf.data(), lengths.data(), offsets.data(), cdf.shape(0),
          cdf.shape(1)};
}
} // namespace

py::list encode_batch(
    const py::list &symbol_arrays, const py::list &index_arrays,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &cdf,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast>
        &lengths,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast>
        &offsets,
    int workers) {
  if (symbol_arrays.size() != index_arrays.size()) {
    throw std::invalid_argument("symbols and indexes task counts differ");
  }
  std::vector<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>
      symbols, indexes;
  std::vector<TaskView> tasks;
  symbols.reserve(symbol_arrays.size());
  indexes.reserve(index_arrays.size());
  tasks.reserve(symbol_arrays.size());
  for (ssize_t i = 0; i < symbol_arrays.size(); ++i) {
    symbols.emplace_back(py::array_t<int32_t, py::array::c_style |
                                     py::array::forcecast>::ensure(
        symbol_arrays[i]));
    indexes.emplace_back(py::array_t<int32_t, py::array::c_style |
                                     py::array::forcecast>::ensure(
        index_arrays[i]));
    if (!symbols.back() || !indexes.back() ||
        symbols.back().size() != indexes.back().size()) {
      throw std::invalid_argument("Invalid symbol/index task");
    }
    tasks.push_back(
        {symbols.back().data(), indexes.back().data(), symbols.back().size()});
  }
  const CdfView cdf_view = make_cdf_view(cdf, lengths, offsets);
  std::vector<std::string> streams(tasks.size());
  workers = std::max(1, std::min(workers, static_cast<int>(tasks.size())));
  {
    py::gil_scoped_release release;
#pragma omp parallel for schedule(static) num_threads(workers)
    for (ssize_t i = 0; i < static_cast<ssize_t>(tasks.size()); ++i) {
      streams[static_cast<size_t>(i)] = encode_one(tasks[i], cdf_view);
    }
  }
  py::list result;
  for (const auto &stream : streams) {
    result.append(py::bytes(stream));
  }
  return result;
}

py::list decode_batch(
    const py::list &stream_list, const py::list &index_arrays,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &cdf,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast>
        &lengths,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast>
        &offsets,
    int workers) {
  if (stream_list.size() != index_arrays.size()) {
    throw std::invalid_argument("streams and indexes task counts differ");
  }
  std::vector<std::string> streams;
  std::vector<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>
      indexes;
  std::vector<IndexView> tasks;
  for (ssize_t i = 0; i < stream_list.size(); ++i) {
    streams.push_back(py::cast<std::string>(stream_list[i]));
    indexes.emplace_back(py::array_t<int32_t, py::array::c_style |
                                     py::array::forcecast>::ensure(
        index_arrays[i]));
    if (!indexes.back()) {
      throw std::invalid_argument("Invalid index task");
    }
    tasks.push_back({indexes.back().data(), indexes.back().size()});
  }
  const CdfView cdf_view = make_cdf_view(cdf, lengths, offsets);
  std::vector<std::vector<int32_t>> outputs(tasks.size());
  workers = std::max(1, std::min(workers, static_cast<int>(tasks.size())));
  {
    py::gil_scoped_release release;
#pragma omp parallel for schedule(static) num_threads(workers)
    for (ssize_t i = 0; i < static_cast<ssize_t>(tasks.size()); ++i) {
      outputs[static_cast<size_t>(i)] =
          decode_one(streams[static_cast<size_t>(i)], tasks[i], cdf_view);
    }
  }
  py::list result;
  for (const auto &output : outputs) {
    py::array_t<int32_t> array(output.size());
    std::copy(output.begin(), output.end(), array.mutable_data());
    result.append(std::move(array));
  }
  return result;
}

PYBIND11_MODULE(_parallel_rans, module) {
  module.doc() = "OpenMP multi-stream unbounded rANS with GIL release";
  module.def("encode_batch", &encode_batch, py::arg("symbols"),
             py::arg("indexes"), py::arg("cdf"), py::arg("cdf_lengths"),
             py::arg("offsets"), py::arg("workers") = 4);
  module.def("decode_batch", &decode_batch, py::arg("streams"),
             py::arg("indexes"), py::arg("cdf"), py::arg("cdf_lengths"),
             py::arg("offsets"), py::arg("workers") = 4);
#ifdef _OPENMP
  module.attr("openmp_enabled") = true;
#else
  module.attr("openmp_enabled") = false;
#endif
}
