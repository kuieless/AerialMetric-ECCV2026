import os
import shutil
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ================= 默认配置区域 (也可以通过命令行覆盖) =================

# Default input/output roots (override via --input and --output arguments)
DEFAULT_INPUT_ROOT = ""
DEFAULT_OUTPUT_ROOT = ""

# 3. 你要提取的文件名 (默认是深度图)
TARGET_FILENAME = "depth.npy" 

# 4. 提取后的后缀 (例如 depth.npy 提取出来变成 xxx.npy)
NEW_EXTENSION = ".npy"

# 5. 线程数 (机械硬盘建议 4-8，固态硬盘可以 16-32)
NUM_THREADS = 16

# ===================================================================

class AdaptiveExtractor:
    def __init__(self, input_root, output_root, target_name, new_ext):
        self.input_root = Path(input_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.target_name = target_name
        self.new_ext = new_ext
        self.tasks = []

    def scan(self):
        """
        全自动扫描：寻找所有匹配 target_name 的文件
        逻辑：
           找到: .../Category/Scene/ImageName/depth.npy
           提取为: .../Output/Category/Scene/ImageName.npy
        """
        print(f"🕵️  正在扫描根目录: {self.input_root}")
        print(f"    目标文件: {self.target_name}")

        # 使用 rglob 递归查找所有匹配的文件
        # 这就是“自适应”的核心，不用写死文件夹列表
        all_targets = list(self.input_root.rglob(self.target_name))
        
        self.tasks = []
        for src_path in all_targets:
            # src_path = .../Category/Scene/ImageName/depth.npy
            
            # 1. 获取包含该文件的文件夹 (即 ImageName)
            image_dir = src_path.parent 
            image_stem = image_dir.name # "ImageName"
            
            # 2. 获取场景文件夹 (即 Scene)
            scene_dir = image_dir.parent
            
            # 3. 计算从根目录开始的相对路径 (用于保持目录结构)
            # 例如: input_root 是 /Data/Out
            # scene_dir 是 /Data/Out/Val/Bench/Scene1
            # rel_path 就是 Val/Bench/Scene1
            try:
                rel_scene_path = scene_dir.relative_to(self.input_root)
            except ValueError:
                # 极端情况：如果文件不在 input_root 下（通常不会发生）
                continue
                
            # 4. 构建目标路径
            # dest_dir = Output / Val / Bench / Scene1
            dest_dir = self.output_root / rel_scene_path
            
            # dest_file = Output / Val / Bench / Scene1 / ImageName.npy
            dest_file = dest_dir / (image_stem + self.new_ext)
            
            self.tasks.append({
                'src': src_path,
                'dst_dir': dest_dir,
                'dst_file': dest_file
            })
            
        print(f"✅ 扫描完成: 发现 {len(self.tasks)} 个待提取文件")

    def _worker(self, task):
        """单个文件的处理逻辑"""
        src = task['src']
        dst_dir = task['dst_dir']
        dst_file = task['dst_file']
        
        try:
            # 创建目录 (如果不存在)
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制文件
            # 如果想用软链接节省空间，把 shutil.copy2 改为 os.symlink(src, dst_file)
            shutil.copy2(src, dst_file) 
            return True
        except Exception as e:
            return str(e)

    def run(self):
        if not self.tasks:
            print("⚠️ 没有找到任何文件，请检查路径配置。")
            return

        print(f"🚀 开始提取到: {self.output_root}")
        print(f"⚙️  使用线程数: {NUM_THREADS}")

        success_count = 0
        fail_count = 0
        
        # 使用线程池加速 I/O 操作
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            # 提交任务
            futures = [executor.submit(self._worker, task) for task in self.tasks]
            
            # 进度条
            for future in tqdm(futures, total=len(self.tasks), unit="file"):
                result = future.result()
                if result is True:
                    success_count += 1
                else:
                    fail_count += 1
                    # print(f"错误: {result}") # 需要调试时打开

        print("\n" + "="*40)
        print(f"🎉 提取完成报告")
        print(f"总计扫描: {len(self.tasks)}")
        print(f"✅ 成功:    {success_count}")
        print(f"❌ 失败:    {fail_count}")
        print("="*40 + "\n")

# ================= 命令行入口 =================

if __name__ == "__main__":
    # 为了方便你直接运行，也保留了命令行传参的功能
    # 如果不传参，就使用顶部的 DEFAULT 变量
    parser = argparse.ArgumentParser(description="自适应提取推理结果脚本")
    parser.add_argument('--input', '-i', default=DEFAULT_INPUT_ROOT, help="输入根目录")
    parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT_ROOT, help="输出根目录")
    parser.add_argument('--target', '-t', default=TARGET_FILENAME, help="要提取的文件名 (如 depth.npy, points.exr)")
    parser.add_argument('--ext', '-e', default=NEW_EXTENSION, help="提取后的后缀 (如 .npy, .exr)")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入目录不存在 -> {args.input}")
    else:
        extractor = AdaptiveExtractor(args.input, args.output, args.target, args.ext)
        extractor.scan()
        extractor.run()