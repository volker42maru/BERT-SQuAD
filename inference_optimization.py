# import tensorrt as trt
# import onnx_tensorrt.backend as backend
import onnx
import torch


class OptimizedModel:

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def build_engine_onnx(self, model_file):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network,
                                                                                                     TRT_LOGGER) as parser:
            builder.max_workspace_size = int(5e9) #common.GiB(1)
            # Load the Onnx model and parse it in order to populate the TensorRT network.
            with open(model_file, 'rb') as model:
                # parser.parse returns a bool, and we were not checking it originally.
                if not parser.parse(model.read()):
                    print(parser.get_error(0))
            print(network.num_layers)
            print(network.get_layer(network.num_layers - 1).get_output(0).shape)
            network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))
            return builder.build_cuda_engine(network)

    def optimize(self, use_xla=False, use_fp16=False, use_trt=False, export_onnx=False, use_onnx_runtime=False, onnx_path=None):
        dummy_input = torch.ones([1,384], dtype=torch.long).to(self.device)
        if use_fp16:
            from pytorch_transformers import AdamW
            from apex import amp
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters()],
                 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=0., eps=0.)
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level='O2')
            # self.model = torch.quantization.quantize_dynamic(self.model, dtype=torch.float16)
            # self.model = self.model.half()
        if use_xla:
            # cannot use xla with trt
            with torch.jit.optimized_execution(False):
                self.model = torch.jit.trace(self.model, (dummy_input, dummy_input, dummy_input), check_trace=True)
            # self.model = torch.jit.script(self.model)
            # self.model.save('/home/volker/workspace/gitrepo/BERT-SQuAD/traced_model.pt')
        if export_onnx:
            torch.onnx.export(self.model, (dummy_input, dummy_input, dummy_input), onnx_path, export_params=True,
                              input_names=['input_ids', 'attention_mask', 'token_type_ids'],
                              output_names=['start_logits', 'end_logits'],
                              do_constant_folding=True,
                              opset_version=10,
                              dynamic_axes={'input_ids': {0: 'batch_size_inputs_ids',
                                                          1: 'seq_len'},
                                            'attention_mask': {0: 'batch_size_attention_mask',
                                                          1: 'seq_len'},
                                            'token_type_ids': {0: 'batch_size_token_type_ids',
                                                          1: 'seq_len'}},
                              operator_export_type=torch.onnx.OperatorExportTypes.ONNX)
        if use_onnx_runtime:
            import onnxruntime
            ort_session = onnxruntime.InferenceSession(onnx_path)
            # return onnx session instead of model
            return ort_session
        if use_trt:
            from torch2trt import torch2trt
            import pycuda.driver as cuda
            dummy_input = torch.ones([1, 384], dtype=torch.int32).to(self.device)
            onnx_path = "/home/volker/workspace/gitrepo/BERT-SQuAD/spanbert_qa.onnx"
            # torch.onnx.export(self.model, dummy_input, onnx_path, export_params=True)
            model = onnx.load(onnx_path)
            engine = backend.prepare(model) #, device='CUDA:1')
            # input_data = np.random.random(size=(1, 1)).astype(np.int32)
            # output_data = engine.run(input_data)[0]
            # engine = self.build_engine_onnx(onnx_path)
            # model_trt = torch2trt(self.model, [dummy_input,dummy_input,dummy_input], max_batch_size=1, max_workspace_size=4e9, log_level=trt.Logger.WARNING)

            # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
            # h_input = cuda.pagelocked_empty(engine.get_binding_shape(0).volume(), dtype=np.int32)
            # h_output = cuda.pagelocked_empty(engine.get_binding_shape(1).volume(), dtype=np.int32)
            # # Allocate device memory for inputs and outputs.
            # d_input = cuda.mem_alloc(h_input.nbytes)
            # d_output = cuda.mem_alloc(h_output.nbytes)
            # # Create a stream in which to copy inputs/outputs and run inference.
            # stream = cuda.Stream()
            # with engine.create_execution_context() as context:
            #     # Transfer input data to the GPU.
            #     cuda.memcpy_htod_async(d_input, h_input, stream)
            #     # Run inference.
            #     context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
            #     # Transfer predictions back from the GPU.
            #     cuda.memcpy_dtoh_async(h_output, d_output, stream)
            #     # Synchronize the stream
            #     stream.synchronize()
            # return cuda engine instead of model
            return engine

        return self.model.eval()
