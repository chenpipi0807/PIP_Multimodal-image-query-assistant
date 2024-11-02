import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import numpy as np
import sys
import os
import torchvision
import gradio as gr
import json
from glob import glob

# 获取当前脚本所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建Janus项目的相对路径
janus_path = os.path.join(current_dir, "Janus")

# 添加Janus项目路径
sys.path.append(janus_path)

# 设置模型文件的本地路径
model_path = "./models/Janus-1.3B"

# 加载模型和处理器
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval() if torch.cuda.is_available() else vl_gpt.to(torch.float32).eval()

cuda_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 多模态理解函数
@torch.inference_mode()
def multimodal_understanding(image_path, question, seed=42, top_p=0.95, temperature=0.1):
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 构建对话
    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>\n{question}",
            "images": [image_path],
        },
        {"role": "Assistant", "content": ""},
    ]
    
    # 加载图像
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(cuda_device, dtype=torch.float32)
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False if temperature == 0 else True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer, prepare_inputs

def generate_caption(image, question):
    caption, _ = multimodal_understanding(image, question)
    return caption

def batch_process_images(image_dir, *questions):
    # 将问题列表转换为模板格式
    question_templates = {f"question_{i+1}": {"description": q, "answer": ""} for i, q in enumerate(questions) if q}
    
    # 支持的图像格式，包括大小写
    supported_formats = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff', '*.webp',
                         '*.PNG', '*.JPG', '*.JPEG', '*.BMP', '*.GIF', '*.TIFF', '*.WEBP')
    
    # 获取所有图像文件
    image_files = set()  # 使用集合来避免重复
    for fmt in supported_formats:
        image_files.update(glob(os.path.join(image_dir, fmt)))
    
    # 转换回列表
    image_files = list(image_files)
    
    # 打印获取的文件列表
    print("Found image files:", image_files)
    
    for image_index, image_path in enumerate(image_files):
        image_name = os.path.basename(image_path)
        output_path = os.path.join(image_dir, f"{os.path.splitext(image_name)[0]}.json")
        
        # 初始化结果字典
        results = {}
        
        for question_index, (key, template) in enumerate(question_templates.items()):
            question = template['description']
            print(f"Processing image {image_index + 1}/{len(image_files)}, question {question_index + 1}/{len(question_templates)}")
            answer, _ = multimodal_understanding(image_path, question)
            results[key] = {"description": question, "answer": answer}
        
        # 保存结果到JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # 更新问询模板
    update_question_templates(question_templates)

def add_question_interface(questions):
    questions.append("")
    return questions

def save_questions_interface(questions):
    question_templates = {f"question_{i+1}": {"description": q, "answer": ""} for i, q in enumerate(questions) if q}
    update_question_templates(question_templates)
    return "问询模板已保存。"

# 更新Gradio界面
def update_question_templates(new_templates):
    with open('问询模板.json', 'w', encoding='utf-8') as f:
        json.dump(new_templates, f, ensure_ascii=False, indent=2)

def load_question_templates():
    with open('问询模板.json', 'r', encoding='utf-8') as f:
        return json.load(f)

# 单张图像处理界面
def single_image_interface():
    return gr.Interface(
        fn=generate_caption,
        inputs=[
            gr.components.Image(type="filepath", label="上传图片"),
            gr.components.Textbox(lines=2, placeholder="在此输入您的问题...", label="问题")
        ],
        outputs=gr.components.Textbox(label="生成的描述"),
        title="PIP多模态图像助理",
        description="上传一张图片并提出一个问题以获取描述。"
    )

# 批量图像处理界面
def batch_image_interface():
    default_questions = load_question_templates()
    question_list = [q['description'] for q in default_questions.values()]
    
    with gr.Blocks() as batch_interface:
        image_dir = gr.Textbox(label="图像目录路径", placeholder="输入图像目录路径")
        
        # 根据JSON模板动态创建问题输入框
        questions = [gr.Textbox(label=f"问题{i+1}", placeholder=f"输入问题{i+1}", value=question_list[i]) for i in range(len(question_list))]
        
        # 布局调整
        with gr.Column():
            # 渲染问题输入框
            for question in questions:
                pass  # 组件在创建时已经自动渲染
            
            # 新增一个问题输入框
            new_question = gr.Textbox(label="新增问题", placeholder="输入新问题")
            
            # 按钮：新增问题并保存
            add_and_save_button = gr.Button("新增问题并保存")
        
        # 点击事件
        def add_and_save(*qs, nq=None):
            qs = list(qs)
            if nq:
                qs.append(nq)
            save_questions_interface(qs)
            # 重新加载JSON文件
            updated_questions = load_question_templates()
            updated_question_list = [q['description'] for q in updated_questions.values()]
            # 返回更新后的问题列表
            return updated_question_list + [""] * (len(questions) - len(updated_question_list)), "问询模板已保存。"
        
        add_and_save_button.click(
            fn=add_and_save,
            inputs=[*questions, new_question],
            outputs=[*questions, gr.Textbox(label="保存状态")]
        )
        
        gr.Interface(
            fn=batch_process_images,
            inputs=[image_dir] + questions,
            outputs=gr.Textbox(label="处理结果"),
            title="PIP多模态图像助理",
            description="批量处理图像并根据输入的问题生成结果。"
        )
    
    return batch_interface

# 创建Gradio界面，包含两个分页
iface = gr.TabbedInterface(
    interface_list=[single_image_interface(), batch_image_interface()],
    tab_names=["单张图像处理", "批量图像处理"]
)

# 启动Gradio接口并自动在浏览器中打开
iface.launch(share=True)
