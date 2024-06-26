#pragma once

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/fastertransformer/core/Buffer.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include <memory>

namespace rtp_llm {

class IHandlerImpl {
public:
    IHandlerImpl(const ft::GptInitParameter& params) : params_(params) {}
    virtual ~IHandlerImpl() {}
    virtual absl::Status loadTensor(std::unordered_map<std::string, ft::ConstBufferPtr>& tensors) = 0;
    virtual std::vector<std::string> tensorInfo() = 0;
    virtual absl::StatusOr<std::unique_ptr<GptModelOutputs>> forward(const ModelRequest& model_input, const GptModelOutputs& model_output) const = 0;
protected:
    const ft::GptInitParameter params_;
};

class HandlerBase {
public:
    HandlerBase(const ft::GptInitParameter& params) : params_(params) {}
    virtual ~HandlerBase() {}
    virtual std::vector<std::string> tensorInfo() { return handler_impl_->tensorInfo(); }
    virtual absl::Status loadTensor(std::unordered_map<std::string, ft::ConstBufferPtr>& tensors) { return handler_impl_->loadTensor(tensors); }
    virtual absl::StatusOr<std::unique_ptr<GptModelOutputs>> forward(const ModelRequest& model_input, const GptModelOutputs& model_output) const {return handler_impl_->forward(model_input, model_output); }
protected:
    const ft::GptInitParameter params_;
    std::unique_ptr<IHandlerImpl> handler_impl_;
};
}  // namespace rtp_llm