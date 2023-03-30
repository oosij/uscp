# import list

## SSH 연결
import paramiko


# ssh 이츠에이 수집한 investing.news 해외 종목 뉴스 접근 및 파일 리스트 추출
def ssh_access_day_files(host, port_num, user_id, user_password, remote_dir_path):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=host, port=port_num, username=user_id, password=user_password)
    stdin, stdout, stderr = ssh.exec_command(f'ls {remote_dir_path}')
    file_list = stdout.read()

    target_list = file_list.decode().split('\n')
    target_list = target_list[:-1]

    return target_list, ssh


# 파일 속 텍스트의 내용 추출
def content_extract(remote_file_path, ssh):
    sftp = ssh.open_sftp()
    with sftp.open(remote_file_path) as f:
        content = f.read().decode()
    content_split = content.split('\n\n')
    body = ''
    for i in range(len(content_split)):
        if len(content_split[i]) > 20:
            body = content_split[i]
    body = body.replace('\n', '')
    body = body.split('Sources:')[0].strip()

    #sftp.close()
    #ssh.close()
    return body
