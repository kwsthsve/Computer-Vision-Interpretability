import time
import torch
from captum.attr import IntegratedGradients, visualization, NoiseTunnel, GradientShap
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel, BlipProcessor, \
    BlipForConditionalGeneration, AutoProcessor, AutoModelForCausalLM
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


# HEADER: Image Captioning Interpretability

def main():
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    device = 'cpu'
    tokenizer = None

    # Load pre-trained models, feature_extractors and tokenizers
    # feature_extractor = ViTImageProcessor.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
    # model = VisionEncoderDecoderModel.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
    # tokenizer = GPT2TokenizerFast.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
    # model_id = 1

    feature_extractor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model_id = 2

    # feature_extractor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
    # model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
    # model_id = 2


    # Load and preprocess the image or images
    image_path = 'Path-to-image'
    image = Image.open(image_path)


    # Define the function to get the output logits
    def forward_func(pixel_values, input_ids, model_id):
        if model_id == 1:
            outputs = model(pixel_values=pixel_values, decoder_input_ids=input_ids.to(device))
        else:
            outputs = model(pixel_values=pixel_values, input_ids=input_ids.to(device))

        logits = outputs.logits
        # print(logits.shape)
        return logits[:, -1, :]  # Return the logits for the last token in the sequence

    # Get attention map in the size of input image
    def get_attention_map(pixel_values, input_ids, model_id):
        if model_id == 1:
            outputs = model(pixel_values=pixel_values, decoder_input_ids=input_ids.to(device), output_attentions=True)

            # Extract attention weights
            decoder_attentions = outputs.cross_attentions  # Cross attentions between encoder and decoder
        else:
            outputs = model(pixel_values=pixel_values, input_ids=input_ids.to(device), output_attentions=True)
            # print(outputs.keys())

            # Extract attention weights
            decoder_attentions = outputs.attentions  # Attentions between encoder and decoder

        # Visualize the attention map for the first decoder layer and first head
        attention_map = decoder_attentions[-1][0, :, -1, :].cpu().detach().numpy()  # Shape: (num_heads, num_patches)
        num_heads, num_patches = attention_map.shape

        # Average over heads
        attention_map = attention_map.mean(axis=0)
        # print(f"Attention Map to be reshaped: {attention_map.shape}")

        attention_map = attention_map[1:197].reshape(int(np.sqrt(num_patches - 1)), int(np.sqrt(num_patches - 1)))  # ViT-base has 14x14 patch outputs
        attention_map = attention_map / attention_map.max()  # Normalize attention map

        # Resize attention map to image size
        attention_map = np.array(Image.fromarray(attention_map).resize(image.size, resample=Image.BILINEAR))

        return attention_map

    def visualize_attention_step(attention_map_list):
        cols = 5
        rows = int(len(attention_map_list) / cols) + 1

        fig, axes = plt.subplots(rows, cols, figsize=(16, 7 * len(attention_map_list)))

        for i in range(rows):
            for j in range(cols):

                if i * cols + j < len(attention_map_list):

                    axes[i, j].imshow(attention_map_list[i * cols + j], cmap='viridis', alpha=0.6)  # Overlay attention map
                    axes[i, j].set_title(f'Attention Map n_tokens = {i * cols + j + 1}')
                    axes[i, j].axis('off')
                else:
                    axes[i, j].axis('off')

        plt.tight_layout()
        plt.show()

    # Transform attribution to image
    def attribution2image(attribution):
        return attribution.squeeze().cpu().detach().numpy().transpose(1, 2, 0)

    # Visualize the attribution
    def visualize_result(attribution_image, original_image):
        _ = visualization.visualize_image_attr_multiple(attribution_image,
                                                            original_image,
                                                            ["original_image", "heat_map"],
                                                            ["all", "positive"],
                                                            titles=["Original Image", "Attribution Heat Map"],
                                                            cmap='viridis',
                                                            show_colorbar=True)

    def visualize_step(attribution_list):
        cols = 5
        rows = int(len(attribution_list) / cols) + 1

        fig, axes = plt.subplots(rows, cols, figsize=(16, 7 * len(attribution_list)))

        for i in range(rows):
            for j in range(cols):

                if i * cols + j < len(attribution_list):

                    visualization.visualize_image_attr(attr=attribution2image(attribution_list[i * cols + j]),
                                                   method="heat_map",
                                                   sign="positive",
                                                   title=f"Attribution Heat Map n_tokens = {i * cols + j + 1}",
                                                   cmap='viridis',
                                                   show_colorbar=True,
                                                   plt_fig_axis=(fig, axes[i, j]),
                                                   use_pyplot=False)
                else:
                    axes[i, j].axis('off')

        plt.tight_layout()
        plt.show()


    start_time = time.time()

    # Initialize IntegratedGradients, GradientShap and NoiseTunnel
    integrated_gradients = IntegratedGradients(forward_func)
    noise_tunnel = NoiseTunnel(integrated_gradients)
    gradient_shap = GradientShap(forward_func)

    def generate_caption(model, feature_extractor, image, tokenizer=None):
        model.to(device)

        # Set the model to evaluation mode
        model.eval()
        model.zero_grad()

        if tokenizer is None:
            tokenizer = feature_extractor

        # Preprocess the image
        inputs = feature_extractor(images=image, return_tensors="pt").to(device)

        # Generate the caption, tune num_beams for multiple captions, tune max_length
        with torch.no_grad():
            outputs_ids = model.generate(pixel_values=inputs.pixel_values, max_length=20, num_beams=1)
            outputs_caption = tokenizer.decode(outputs_ids[0], skip_special_tokens=True)

        return outputs_ids, outputs_caption, inputs.pixel_values


    output_ids, output_caption, pixel_values = generate_caption(model, feature_extractor, image, tokenizer)
    output_ids = output_ids.tolist()[0]

    if tokenizer is None:
        tokenizer = feature_extractor

    # Print results
    print(f'Shape of input: {pixel_values.shape}')

    print(f'Generated Vocabulary IDs (with special tokens): {output_ids}\n')
    for i, token in enumerate(output_ids):
        print(
            f'Token {i + 1} / ID {token} = {tokenizer.batch_decode(torch.tensor([[token]]), skip_special_tokens=False)[0]}')
    print(f'\nGenerated Caption: {output_caption}')

    # Move the pixel values to the model's device
    pixel_values = pixel_values.to(device)

    # Rescale original image to [0, 1] for visualization
    original_image = pixel_values.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

    # Compute the attributions
    try:
        attributions = []
        attention_maps = []
        for i in range(len(output_ids)):
            target_sentence = output_ids[:i + 1]

            # torch.manual_seed(0)
            # np.random.seed(0)
            # rand_img_dist = torch.cat([pixel_values * 0, pixel_values * 1])

            # attributions_gs = gradient_shap.attribute(pixel_values, n_samples=50, stdevs=0.0001, baselines=rand_img_dist, target=i, additional_forward_args=(torch.tensor([target_sentence]), model_id))
            attributions_ig = integrated_gradients.attribute(inputs=pixel_values, target=i, n_steps=10, additional_forward_args=(torch.tensor([target_sentence]), model_id))
            # attributions_nt = noise_tunnel.attribute(inputs=pixel_values, nt_samples=5, nt_type='smoothgrad_sq', target=i, additional_forward_args=(torch.tensor([target_sentence]), model_id))

            attributions.append(attributions_ig)
            attention_maps.append(get_attention_map(pixel_values, torch.tensor([target_sentence]), model_id))

        visualize_step(attributions)
        visualize_attention_step(attention_maps)

        combined_attributions = torch.stack(attributions, dim=0)
        visualize_result(attribution2image(torch.sum(combined_attributions, dim=0)), original_image)

    except Exception as e:
        print(f"Error during attribution computation: {e}")
        return

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f'\nElapsed time: {elapsed_time / 60:.2f} minutes')


if __name__ == "__main__":
    main()
