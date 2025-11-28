/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: XSMTAsync.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace xsmt_async {

/// A dialect for async.
/// This dialect contains primitives for async.
class XSMTAsyncDialect : public ::mlir::Dialect {
  explicit XSMTAsyncDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~XSMTAsyncDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("xsmt_async");
  }
};
} // namespace xsmt_async
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xsmt_async::XSMTAsyncDialect)
