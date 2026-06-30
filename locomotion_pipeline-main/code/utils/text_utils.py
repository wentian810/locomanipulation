import os
import re
from sys import prefix

class FileText:
    def __init__(self, file_path):
        self.file_path = file_path
        self.content = self.read().splitlines()

    def read(self):
        with open(self.file_path, 'r') as f:
            return f.read()

    def write(self, content):
        with open(self.file_path, 'w') as f:
            f.write(content)

    def filter_lines(self, pattern):
        filtered_lines = [line for line in self.content if re.search(pattern, line)]
        return filtered_lines
    
    def operate_lines(self, op_type, **kwargs):
        out = []
        if op_type == 'seg':
            for line in self.content:
                segs = line.split(kwargs['sep'])
                start = kwargs.get('start', 0)
                end = kwargs.get('end', len(segs))
                out.append(kwargs['sep'].join(segs[start:end]))
        elif op_type == 'replace':
            for line in self.content:
                out.append(line.replace(kwargs['old'], kwargs['new']))
        else:
            raise ValueError(f"Unsupported operation type: {op_type}")
        return out
    
    @staticmethod
    def save_text(file_path, lines):
        with open(file_path, 'w') as f:
            f.write('\n'.join(lines))

if __name__ == "__main__":
    file_text_folders = FileText('log/tmp_files/folder_list.txt')
    file_text_videos = FileText('log/tmp_files/video_list.txt')

    video_prefixes = file_text_videos.operate_lines('seg', sep='_', end=1)
    folder_prefixes = file_text_folders.operate_lines('seg', sep='_', end=1)

    filter_lines = []
    for video_idx, prefix_video in enumerate(video_prefixes):
        prefix_video_int = int(prefix_video)
        for folder_idx, prefix_folder in enumerate(folder_prefixes):
            prefix_folder_int = int(prefix_folder)
            if prefix_video_int == prefix_folder_int:
                print(f"Match found: Video prefix '{prefix_video}' matches Folder prefix '{prefix_folder}'")
                filter_lines.append(f'{file_text_folders.content[folder_idx]}/{file_text_videos.content[video_idx]}.mp4')
                break
            elif prefix_video_int < prefix_folder_int:
                break
    # print(filter_lines)
    FileText.save_text('log/tmp_files/filter_lines.txt', filter_lines)

    for line in filter_lines:
        remote_video_path = f'batch1/姿态转换/{line}'
        os.system(
            f'bash code/scripts/run_fsys_sync.sh --regex-pattern {remote_video_path}'
        )
