# HQQ

ينفذ التكميم نصف الرباعي (HQQ) التكميم أثناء الطيران عبر التحسين السريع والقوي. لا يتطلب بيانات المعايرة ويمكن استخدامه لتكميم أي نموذج. يرجى الرجوع إلى <a href="https://github.com/mobiusml/hqq/">الحزمة الرسمية</a> لمزيد من التفاصيل.

للتركيب، نوصي باستخدام النهج التالي للحصول على أحدث إصدار وبناء نواة CUDA المقابلة:

```
pip install hqq
```

لتكميم نموذج، تحتاج إلى إنشاء [`HqqConfig`]. هناك طريقتان للقيام بذلك:

```Python
from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig

# الطريقة 1: ستستخدم جميع الطبقات الخطية نفس تكوين التكميم
quant_config = HqqConfig(nbits=8, group_size=64, quant_zero=False, quant_scale=False, axis=0) #يتم استخدام axis=0 بشكل افتراضي
```

```Python
# الطريقة 2: ستستخدم كل طبقة خطية بنفس العلامة تكوين تكميم مخصص
q4_config = {'nbits':4, 'group_size':64, 'quant_zero':False, 'quant_scale':False}
q3_config = {'nbits':3, 'group_size':32, 'quant_zero':False, 'quant_scale':False}
quant_config  = HqqConfig(dynamic_config={
'self_attn.q_proj':q4_config,
'self_attn.k_proj':q4_config,
'self_attn.v_proj':q4_config,
'self_attn.o_proj':q4_config,

'mlp.gate_proj':q3_config,
'mlp.up_proj'  :q3_config,
'mlp.down_proj':q3_config,
})
```

النهج الثاني مثير للاهتمام بشكل خاص لتكميم مزيج الخبراء (MoEs) لأن الخبراء أقل تأثرًا بإعدادات التكميم المنخفضة.

بعد ذلك، قم ببساطة بتكميم النموذج كما يلي:

```Python
model = transformers.AutoModelForCausalLM.from_pretrained(
model_id,
torch_dtype=torch.float16,
device_map="cuda",
quantization_config=quant_config
)
```

## وقت التشغيل الأمثل

يدعم HQQ عدة واجهات خلفية، بما في ذلك PyTorch النقي ونوى CUDA المخصصة لإزالة التكميم. هذه الواجهات الخلفية مناسبة لمعالجات الرسوميات الأقدم وللتدريب peft/QLoRA.

للحصول على استدلال أسرع، يدعم HQQ نوى مدمجة من 4 بت (TorchAO وMarlin)، حيث تصل إلى 200 رمز/ثانية على 4090 واحدة.

لمزيد من التفاصيل حول كيفية استخدام الواجهات الخلفية، يرجى الرجوع إلى https://github.com/mobiusml/hqq/?tab=readme-ov-file#backend