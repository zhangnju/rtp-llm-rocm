#include "src/fastertransformer/models/multi_gpu_gpt/NormWrapper.h"
#include "src/fastertransformer/kernels/rmsnormKernels.h"

namespace fastertransformer
{
template<typename T>
void NormWrapper<T>::initDecoderLayerNorm(T* output, 
                                          const T*     input, 
                                          const T*     gamma, 
                                          const T*     beta, 
                                          const float  eps,
                                          const size_t m,
                                          const size_t n,
                                          // scale and sq_int8 only used when layernorm
                                          float*       scale,
                                          float*       dynamic_scale,
                                          const bool   sq_int8,
                                          int8_t*      norm_output_quant,
                                          cudaStream_t stream) 
{
    if (sq_int8) {
        if (layernorm_type_ == LayerNormType::pre_layernorm) {
            if (norm_type_ == NormType::rmsnorm) {
                invokeGeneralRmsNorm((T*)nullptr, input, gamma, beta, eps, m, n, stream, scale, dynamic_scale, norm_output_quant);
            } else {
                invokeGeneralLayerNorm((T*)nullptr, input, gamma, beta, eps, m, n, stream, true, scale, dynamic_scale, norm_output_quant);
            }
        }

    } else {
        if (layernorm_type_ == LayerNormType::pre_layernorm) {
            if (norm_type_ == NormType::rmsnorm) {
                invokeGeneralRmsNorm(output, input, gamma, beta, eps, m, n, stream);
            } else {
                invokeGeneralLayerNorm(output, input, gamma, beta, eps, m, n);
            }
        }
    }
}

template<typename T>
void NormWrapper<T>::preAttentionLayerNorm(T*           output,
                                           const T*     input,
                                           const T*     gamma,
                                           const T*     beta,
                                           const float  eps,
                                           const size_t m,
                                           const size_t n,
                                           const float* scale,
                                           float*       dynamic_scale,
                                           const bool   sq_int8,
                                           int8_t*      norm_output_quant,
                                           cudaStream_t stream) {
    if (sq_int8) {
        if (layernorm_type_ == LayerNormType::pre_layernorm) {
            invokeGeneralLayerNorm((T*)nullptr, input, gamma, beta, eps, m, n, stream, true, scale, dynamic_scale, norm_output_quant);
        }

    } else {
        if (layernorm_type_ == LayerNormType::pre_layernorm) {
            invokeGeneralLayerNorm(output, input, gamma, beta, eps, m, n);
        }
    }
}

template<typename T>
void NormWrapper<T>::attentionAddBiasResidualLayerNorm(T*           output,
                                                       T*           norm_output,
                                                       const T*     input,
                                                       const T*     residual1,
                                                       const T*     gamma,
                                                       const T*     beta,
                                                       const T*     bias,
                                                       const float  eps,
                                                       const size_t m,
                                                       const size_t n,
                                                       const float* scale,
                                                       float*       dynamic_scale,
                                                       const bool   sq_int8,
                                                       int8_t*      norm_output_quant,
                                                       cudaStream_t stream) {
    if (sq_int8) {
        if (layernorm_type_ == LayerNormType::pre_layernorm) {
            if (norm_type_ == NormType::layernorm) {
                // norm_output = layernorm(input + residual1 + residual2 + bias) * gamma + beta
                invokeGeneralAddBiasResidualLayerNorm(
                    output, (T*)nullptr, input, bias, residual1, gamma, beta, eps, m, n, stream, true, scale, dynamic_scale, norm_output_quant);
            } else if (norm_type_ == NormType::rmsnorm) {
                // norm_output = rmsnorm(output + residual1 + bias) * gamma + beta
                invokeAddBiasResidualRmsNorm(
                    output, (T*)nullptr, input, bias, residual1, gamma, beta, eps, m, n, stream, scale, dynamic_scale, norm_output_quant);
            }
        }

    } else {
        if (layernorm_type_ == LayerNormType::pre_layernorm) {
            if (norm_type_ == NormType::layernorm) {
                // norm_output = layernorm(input + residual1 + residual2 + bias) * gamma + beta
                invokeGeneralAddBiasResidualLayerNorm(
                    output, norm_output, input, bias, residual1, gamma, beta, eps, m, n);
            } else if (norm_type_ == NormType::alphanorm) {
                // norm_output = layernorm(input + residual1 * alpha + bias) * gamma + beta
                // won't support W8A8 since it's only implemented in ChatGLM
                invokeAlphaAddBiasResidualLayerNorm(
                    norm_output, input, residual1, bias, gamma, beta, alpha_, m, n, stream);
            } else if (norm_type_ == NormType::rmsnorm) {
                // norm_output = rmsnorm(output + residual1 + bias) * gamma + beta
                invokeAddBiasResidualRmsNorm(
                    output, norm_output, input, bias, residual1, gamma, beta, eps, m, n, stream);
            }
        } else if (layernorm_type_ == LayerNormType::post_layernorm) {
            // output = layernorm(output + residual1 + bias) * gamma + beta
            // won't support W8A8 since it's only implemented in BERT
            invokeAddBiasResidualLayerNorm(output, residual1, bias, gamma, beta, eps, m, n, stream);
        }
    }
}

template<typename T>
void NormWrapper<T>::ffnAddBiasResidualLayerNorm(T*           output,
                                                 const T*     input,
                                                 const T*     residual1,
                                                 const T*     residual2,
                                                 const T*     bias,
                                                 const T*     gamma,
                                                 const T*     beta,
                                                 const float  eps,
                                                 const size_t m,
                                                 const size_t n,
                                                 const float* scale_inter,
                                                 const float* scale_out,
                                                 cudaStream_t stream)
{
    if (layernorm_type_ == LayerNormType::pre_layernorm) {
        if (norm_type_ == NormType::alphanorm) {
            // output = input * alpha + residual1 + bias 
            invokeAlphaAddBiasResidual(output,
                                       input,
                                       residual1,
                                       bias,
                                       alpha_,
                                       m,
                                       n,
                                       stream);
        } else {
            // layernorm or rmsnorm
            // output = input + residual1 + residual2 + bias
            invokeAddBiasResidual(output,
                                  input,
                                  residual1,
                                  residual2,
                                  bias,
                                  scale_inter,
                                  scale_out,
                                  m,
                                  n,
                                  stream);
        }
    } else if (layernorm_type_ == LayerNormType::post_layernorm) {
        invokeAddBiasResidualLayerNorm(output,
                                       input,
                                       bias,
                                       gamma,
                                       beta,
                                       eps,
                                       m,
                                       n,
                                       stream);
    }
}

template class NormWrapper<float>;
template class NormWrapper<half>;
#ifdef ENABLE_BF16
template class NormWrapper<__nv_bfloat16>;
#endif
    
} // namespace fastertransformer
