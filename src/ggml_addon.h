//
// Created by geraltigas on 2/26/24.
//

#ifndef GGML_ADDON_H
#define GGML_ADDON_H

#include <ggml_addon.h>
#include <ggml.h>
#include <sstream>
#include <string>

void replace_all(std::string & s, const std::string & search, const std::string & replace);
std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i);

#endif //GGML_ADDON_H
