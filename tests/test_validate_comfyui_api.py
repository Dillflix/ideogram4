from __future__ import annotations

import unittest

from scripts.validate_comfyui_api import _saved_images, _workflow


class ComfyUIValidationTests(unittest.TestCase):
  def test_workflow_uses_local_caption_and_acceptance_settings(self) -> None:
    caption = '{"high_level_description":"test"}'

    workflow = _workflow(caption)

    self.assertEqual(workflow["1"]["class_type"], "Ideogram4PipelineLoader")
    self.assertEqual(workflow["1"]["inputs"]["model_weights"], "4.0 NF4")
    self.assertEqual(workflow["2"]["class_type"], "Ideogram4Generate")
    self.assertEqual(workflow["2"]["inputs"]["pipeline"], ["1", 0])
    self.assertEqual(workflow["2"]["inputs"]["prompt"], caption)
    self.assertEqual(workflow["2"]["inputs"]["width"], 1024)
    self.assertEqual(workflow["2"]["inputs"]["height"], 1024)
    self.assertEqual(workflow["2"]["inputs"]["sampler_preset"], "4.0 Default 20")
    self.assertEqual(workflow["3"]["class_type"], "SaveImage")
    self.assertEqual(workflow["3"]["inputs"]["images"], ["2", 0])

  def test_saved_images_reads_save_node_output(self) -> None:
    expected = [{"filename": "result.png", "subfolder": "", "type": "output"}]

    images = _saved_images({"outputs": {"3": {"images": expected}}})

    self.assertEqual(images, expected)


if __name__ == "__main__":
  unittest.main()
