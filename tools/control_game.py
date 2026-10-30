#!/usr/bin/env python3
"""Bounded local input for one explicitly enabled native test process."""
from pathlib import Path
import argparse,time,struct,math
p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--buttons',default='');p.add_argument('--hold-ms',type=int,default=160);p.add_argument('--sticks',nargs=4,type=float,default=[0,0,0,0]);p.add_argument('--frame',action='store_true');p.add_argument('--wait-frame',action='store_true');p.add_argument('--present-interval',type=int,choices=[0,1,2]);p.add_argument('--present-duration-ms',type=int,default=120000);p.add_argument('--inspect-config',action='store_true');p.add_argument('--inspect-frame-settings',action='store_true');p.add_argument('--set-frame-field',choices=['sim-max-fps','sim-force-rate','display-interval']);p.add_argument('--object-address',type=lambda s:int(s,0));p.add_argument('--expected');p.add_argument('--value');p.add_argument('--edit-duration-ms',type=int,default=45000);p.add_argument('--restore-frame-settings',action='store_true');p.add_argument('--buffer-formats-mode',type=int,choices=[0,1]);p.add_argument('--profile-frame-waits',type=int);p.add_argument('--inspect-engine-limiter',action='store_true');p.add_argument('--set-engine-fps',type=int,choices=[30,60]);p.add_argument('--expected-engine-fps',type=int,choices=[30,60],default=30);p.add_argument('--restore-engine-fps',action='store_true');p.add_argument('--inspect-render-quality',action='store_true');p.add_argument('--set-render-field',choices=['mesh-force-lod','highres-crowd-fan','pitch-clumps','crowd-selection-max','highres-sideline','highres-sideline-max','resolution-scale','resolution-scale-min','dynamic-resolution','force-dof','dynamic-ao','motion-blur','filmic-effects','subsurface','mesh-lod-scale','crowd-lod-scale','crowd-boost-lod-scale']);p.add_argument('--restore-render-quality',action='store_true');p.add_argument('--cpu-texture-checks-mode',type=int,choices=[0,1]);a=p.parse_args()
assert 1<=a.hold_ms<=5000 and all(-1<=v<=1 for v in a.sticks)
names=['A','B','X','Y','LSTICK','RSTICK','L','R','ZL','ZR','PLUS','MINUS','LEFT','UP','RIGHT','DOWN'];mask=0
for name in filter(None,a.buttons.upper().split('+')):mask|=1<<names.index(name)
sequence=time.time_ns();path=a.run/'test-control'
if a.cpu_texture_checks_mode is not None:
 assert not a.buttons and not any(a.sticks) and not a.frame and 1000<=a.edit_duration_ms<=120000
 request=path.with_suffix('.cpu-texture-checks');temp=request.with_suffix('.tmp');temp.write_text(f'{sequence} {a.cpu_texture_checks_mode} {a.edit_duration_ms}\n');temp.replace(request);print(path.with_suffix(f'.cpu-texture-checks-{sequence}.json'))
if a.buffer_formats_mode is not None:
 assert not a.buttons and not any(a.sticks) and not a.frame and 1000<=a.edit_duration_ms<=120000
 request=path.with_suffix('.buffer-formats');temp=request.with_suffix('.tmp');temp.write_text(f'{sequence} {a.buffer_formats_mode} {a.edit_duration_ms}\n');temp.replace(request);print(path.with_suffix(f'.buffer-formats-{sequence}.json'))
if a.inspect_render_quality or a.set_render_field or a.restore_render_quality:
 assert sum([a.inspect_render_quality,bool(a.set_render_field),a.restore_render_quality])==1 and not a.buttons and not any(a.sticks) and a.present_interval is None and not a.set_frame_field and not a.restore_frame_settings and not a.inspect_config and not a.inspect_frame_settings and a.profile_frame_waits is None and not a.inspect_engine_limiter and a.set_engine_fps is None and not a.restore_engine_fps
 if a.inspect_render_quality:command=f'read {sequence}\n';output=path.with_suffix(f'.render-{sequence}.json')
 elif a.restore_render_quality:command=f'reset {sequence}\n';output=path.with_suffix(f'.render-edit-{sequence}.json')
 else:
  assert a.object_address and a.expected is not None and a.value is not None and 1000<=a.edit_duration_ms<=120000
  if a.set_render_field in ['resolution-scale','resolution-scale-min','mesh-lod-scale','crowd-lod-scale','crowd-boost-lod-scale']:
   values=[float(a.expected),float(a.value)];assert all(math.isfinite(v) and (0 if a.set_render_field=="resolution-scale-min" else .125)<=v<=4 for v in values);old,new=[struct.unpack('<i',struct.pack('<f',v))[0] for v in values]
  else:old,new=int(a.expected),int(a.value)
  command=f'edit {sequence} {a.set_render_field} {a.object_address:#x} {old} {new} {a.edit_duration_ms}\n';output=path.with_suffix(f'.render-edit-{sequence}.json')
 request=path.with_suffix('.render-quality');temp=request.with_suffix('.tmp');temp.write_text(command);temp.replace(request);print(output)
if a.inspect_engine_limiter or a.set_engine_fps is not None or a.restore_engine_fps:
 assert sum([a.inspect_engine_limiter,a.set_engine_fps is not None,a.restore_engine_fps])==1 and not a.buttons and not any(a.sticks) and a.present_interval is None and not a.set_frame_field and not a.restore_frame_settings and not a.inspect_config and not a.inspect_frame_settings and a.profile_frame_waits is None
 if a.inspect_engine_limiter:command=f'read {sequence}\n'
 elif a.restore_engine_fps:command=f'reset {sequence}\n'
 else:
  assert 100<=a.edit_duration_ms<=120000
  command=f'edit {sequence} {a.expected_engine_fps} {a.set_engine_fps} {a.edit_duration_ms}\n'
 request=path.with_suffix('.limiter');temp=request.with_suffix('.tmp');temp.write_text(command);temp.replace(request)
 print(path.with_suffix(f'.limiter-{sequence}.json'))
if a.profile_frame_waits is not None:
 assert 1000<=a.profile_frame_waits<=5000 and not a.buttons and not any(a.sticks) and a.present_interval is None and not a.set_frame_field and not a.restore_frame_settings and not a.inspect_config and not a.inspect_frame_settings
 request=path.with_suffix('.wait-profile');temp=request.with_suffix('.tmp');temp.write_text(f'{sequence} {a.profile_frame_waits}\n');temp.replace(request)
 print(path.with_suffix(f'.wait-profile-{sequence}.json'))
if a.set_frame_field or a.restore_frame_settings:
 assert not a.buttons and not any(a.sticks) and a.present_interval is None and not a.inspect_config and not a.inspect_frame_settings
 if a.restore_frame_settings:
  assert not a.set_frame_field
  command=f'reset {sequence}\n'
 else:
  assert a.object_address and a.object_address%8==0 and a.expected is not None and a.value is not None and 100<=a.edit_duration_ms<=120000
  command=f'edit {sequence} {a.set_frame_field} {a.object_address:#x} {a.expected} {a.value} {a.edit_duration_ms}\n'
 request=path.with_suffix('.setting-edit');temp=request.with_suffix('.tmp');temp.write_text(command);temp.replace(request)
 print(path.with_suffix(f'.edit-{sequence}.json'))
if a.inspect_frame_settings:
 assert not a.buttons and not any(a.sticks) and a.present_interval is None and not a.inspect_config
 request=path.with_suffix('.settings');temp=request.with_suffix('.tmp');temp.write_text(str(sequence));temp.replace(request)
 print(path.with_suffix(f'.settings-{sequence}.json'))
if a.inspect_config:
 assert not a.buttons and not any(a.sticks) and a.present_interval is None
 request=path.with_suffix('.inspect')
 temp=request.with_suffix('.tmp');temp.write_text(str(sequence));temp.replace(request)
 print(path.with_suffix(f'.inspect-{sequence}.json'))
if a.present_interval is not None:
 assert not a.buttons and not any(a.sticks) and 100<=a.present_duration_ms<=120000
 temp=path.with_suffix('.tmp');temp.write_text(f'present {sequence} {a.present_interval} {a.present_duration_ms}\n');temp.replace(path)
 print(f'presentation sequence {sequence}, interval {a.present_interval}, expires after {a.present_duration_ms}ms; guest clocks unchanged')
if a.buttons or any(a.sticks):
 temp=path.with_suffix('.tmp');temp.write_text(f'{sequence} {mask:x} {a.hold_ms} '+ ' '.join(map(str,a.sticks))+'\n');temp.replace(path)
 print(f'input sequence {sequence}, buttons {a.buttons}, expires after {a.hold_ms}ms')
if a.frame:
 request=path.with_suffix('.frame');temp=request.with_suffix('.tmp');temp.write_text(str(sequence));temp.replace(request)
 output=a.run/'work/frame'/f'manual-{sequence:06}.png'
 if a.wait_frame:
  deadline=time.monotonic()+20
  while True:
   try:
    with output.open('rb') as f:f.seek(-12,2);complete=f.read()==b'\x00\x00\x00\x00IEND\xaeB\x60\x82'
   except (OSError,ValueError):complete=False
   if complete:break
   if time.monotonic()>=deadline:raise TimeoutError(f'no complete framebuffer: {output}')
   time.sleep(0.05)
 print(output)
