# Copyright (c) 2020-201, NVIDIA CORPORATION. All rights reserved.

import numpy as np
import sys
import json
from pathlib import Path

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.


import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config  = json.loads(args['model_config'])
        self.model_instance_device_id  = json.loads(args['model_instance_device_id'])

        import numba.cuda as cuda
        cuda.select_device(self.model_instance_device_id)
        import cudf
        from cudf.core.subword_tokenizer import SubwordTokenizer

        # get vocab
        v_p = Path(__file__).with_name('vocab_hash.txt')
        self.cudf_tokenizer = SubwordTokenizer(v_p, do_lower_case=True)

        self.cudf_lib = cudf
        self.seq_len = 256

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            raw_strings = pb_utils.get_input_tensor_by_name(request, "product_reviews").as_numpy()
            str_ls = [s.decode() for s in raw_strings]
            str_series = self.cudf_lib.Series(str_ls)
            
            ### Use RAPIDS
            cudf_output = self.cudf_tokenizer(str_series,
                    max_length=self.seq_len,
                    max_num_rows=len(str_series),
                    padding="max_length",
                    return_tensors="cp",
                    truncation=True,
                    add_special_tokens=False)
    
            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            
            ### Wont need .get() conversion in newer releases
            ### see PR https://github.com/triton-inference-server/python_backend/pull/62
            input_ids = cudf_output['input_ids'].astype(np.int32).toDlpack()
            attention_mask = cudf_output['attention_mask'].astype(np.int32).toDlpack()
            metadata = cudf_output['metadata'].astype(np.int32).toDlpack()
            

            out_tensor_0 = pb_utils.Tensor.from_dlpack("input_ids", input_ids)
            out_tensor_1 = pb_utils.Tensor.from_dlpack("attention_mask", attention_mask )
            out_tensor_2 = pb_utils.Tensor.from_dlpack("metadata", metadata)
            

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1,out_tensor_2])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
