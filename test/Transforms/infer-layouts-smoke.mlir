// RUN: frisk-opt %s -frisk-infer-layouts | FileCheck %s

module {
  func.func @smoke() {
    return
  }
}

// CHECK: func.func @smoke
// CHECK-NOT: frisk.layout_inference_ran
