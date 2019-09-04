function varargout = signal1D(varargin)
% SIGNAL1D M-file for signal1D.fig
%      SIGNAL1D, by itself, creates a new SIGNAL1D or raises the existing
%      singleton*.
%
%      H = SIGNAL1D returns the handle to a new SIGNAL1D or the handle to
%      the existing singleton*.
%
%      SIGNAL1D('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SIGNAL1D.M with the given input arguments.
%
%      SIGNAL1D('Property','Value',...) creates a new SIGNAL1D or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before signal1D_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to signal1D_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help signal1D

% Last Modified by GUIDE v2.5 23-Jun-2015 17:17:54

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @signal1D_OpeningFcn, ...
                   'gui_OutputFcn',  @signal1D_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

function signal1D_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to signal1D (see VARARGIN)

% Choose default command line output for signal1D
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

tbshan_open_Callback(hObject, eventdata, handles);

% on ajoute le path pour que l interface continue meme si on change le
% dossier
p=mfilename('fullpath');
[pathstr, name, ext] = fileparts(p);

set(findobj('Tag','tbtfd_ouvrir1'),'CData',iconRead(fullfile(pathstr, 'opendoc.mat')));
set(findobj('Tag','tb1_ouvrir1'),'CData',iconRead(fullfile(pathstr, 'opendoc.mat')));
set(findobj('Tag','tb1_ouvrir2'),'CData',iconRead(fullfile(pathstr, 'opendoc.mat')));
set(findobj('Tag','tb2_ouvrir1'),'CData',iconRead(fullfile(pathstr, 'opendoc.mat')));
set(findobj('Tag','tb2_ouvrir2'),'CData',iconRead(fullfile(pathstr, 'opendoc.mat')));
set(findobj('Tag','tb1_sauvegarder_sortie'),'CData',iconRead(fullfile(pathstr, 'savedoc.mat')));
set(findobj('Tag','tb2_sauvegarder_sortie'),'CData',iconRead(fullfile(pathstr, 'savedoc.mat')));
set(findobj('Tag','tb4_sauvegarder'),'CData',iconRead(fullfile(pathstr, 'savedoc.mat')));
set(findobj('Tag','tb3_sauvegarder'),'CData',iconRead(fullfile(pathstr, 'savedoc.mat')));
set(findobj('Tag','tb3_ouvrir'),'CData',iconRead(fullfile(pathstr, 'opendoc.mat')));
set(findobj('Tag','tb4_ouvrir'),'CData',iconRead(fullfile(pathstr, 'opendoc.mat')));

axes(handles.tb3_poles)
cla;
axes(handles.tb3_ri)
cla;
axes(handles.tb3_rf)
cla;
axes(handles.tb4_poles)
cla;
axes(handles.tb4_ri)
cla;
axes(handles.tb4_rf)
cla;

set(findobj('Tag','tb3_bi'),'String','[]');
set(findobj('Tag','tb4_bi'),'String','[]');
set(findobj('Tag','tb4_ai'),'String','[]');

%echant_ButtonDownFcn(hObject, eventdata, handles);
addpath( pathstr );

function varargout = signal1D_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

function tb1_ouvrir1_Callback(hObject, eventdata, handles)
% hObject    handle to tb2_ouvrir1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[FileName,PathName] = uigetfile('*.mat','Choisir un signal (mat)');
if (~ischar(FileName) && FileName == 0) || (~ischar(PathName) && PathName == 0)
    return;
end

in=load([PathName FileName]);
axes(handles.tb1_in1_t)
str=fieldnames(in);
eval(['signal_X_tb1=in.' str{1} ';']);
h=stem(signal_X_tb1);
set(h,'Tag','signal_X_tb1')

% signal de sortie
h=findobj('Tag','signal_Y_tb1');
if ~isempty(h)
    signal_Y_tb1=get(h,'YData');
    signal_sortie_tb1=xcorr(signal_X_tb1,signal_Y_tb1);
    axes(handles.tb1_out_t)
    h=stem(abs(signal_sortie_tb1));
    set(h,'Tag','signal_sortie_tb1')
end

function tb1_ouvrir2_Callback(hObject, eventdata, handles)
% hObject    handle to tb2_ouvrir2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[FileName,PathName] = uigetfile('*.mat','Choisir un signal (mat)');
if (~ischar(FileName) && FileName == 0) || (~ischar(PathName) && PathName == 0)
    return;
end

in=load([PathName FileName]);
axes(handles.tb1_in2_t)
str=fieldnames(in);
eval(['signal_Y_tb1=in.' str{1} ';']);
h=stem(signal_Y_tb1);
set(h,'Tag','signal_Y_tb1')

% signal de sortie
h=findobj('Tag','signal_X_tb1');
if ~isempty(h)
    signal_X_tb1=get(h,'YData');
    signal_sortie_tb1=xcorr(signal_X_tb1,signal_Y_tb1);
    axes(handles.tb1_out_t)
    h=stem(abs(signal_sortie_tb1));
    set(h,'Tag','signal_sortie_tb1')
end

function tb1_sauvegarder_sortie_Callback(hObject, eventdata, handles)
% hObject    handle to tb2_sauvegarder_sortie (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

h=findobj('Tag','signal_sortie_tb1');
if ~isempty(h)
    signal_correlation=get(h,'YData');
else
    signal_correlation=[];
end
[file,path] = uiputfile('*.mat','Sauvegarder le signal de corrï¿½ation');
if (~ischar(file) && file == 0) || (~ischar(path) && path == 0)
    return;
end
save([path file],'signal_correlation');

function tb2_ouvrir1_Callback(hObject, eventdata, handles)
% hObject    handle to tb2_ouvrir1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[FileName,PathName] = uigetfile('*.mat','Choisir un signal (mat)');
if (~ischar(FileName) && FileName == 0) || (~ischar(PathName) && PathName == 0)
    return;
end

in=load([PathName FileName]);
axes(handles.tb2_entree_t)
str=fieldnames(in);
eval(['signal_entree_tb2=in.' str{1} ';']);
h=plot(signal_entree_tb2);
set(h,'Tag','signal_entree_tb2')
xlabel('t');
ylabel('x(t)');

signal_fft=fftshift(fft(signal_entree_tb2));
axes(handles.tb2_entree_f)

xf=-0.5:1/length(signal_entree_tb2):0.5;
xf=xf(1:length(signal_entree_tb2));
cb=findobj('Tag','tb2_checkbox1');
if ~isempty(cb)
    val=get(cb,'Value');
else
    val=0;
end
if val==1
    plot(xf,20*log10(abs(signal_fft)));
else
    plot(xf,abs(signal_fft));
end
xlabel('w/we');
ylabel('|X(w/we)|');
set(handles.tb2_entree_f,'Xtick', -1/2:0.25:1/2);
% signal de sortie

h=findobj('Tag','signal_filtre_tb2');
if ~isempty(h)
    signal_filtre_tb2=get(h,'YData');
    signal_sortie=filter(signal_filtre_tb2,1,signal_entree_tb2);
    axes(handles.tb2_sortie_t)
    h=plot(abs(signal_sortie));
    set(h,'Tag','signal_sortie_tb2')
    xlabel('t');
    ylabel('x(t)');
    signal_sortie_fft=fftshift(fft(signal_sortie));
    axes(handles.tb2_sortie_f)
    xf=-0.5:1/length(signal_sortie):0.5;
    xf=xf(1:length(signal_sortie));

    cb=findobj('Tag','tb2_checkbox1');
    if ~isempty(cb)
        val=get(cb,'Value');
    else
        val=0;
    end
    if val==1
        plot(xf,20*log10(abs(signal_sortie_fft)));
    else
        plot(xf,abs(signal_sortie_fft));
    end
    xlabel('w/we');
    ylabel('|X(w/we)|');
    set(handles.tb2_sortie_f,'Xtick', -1/2:0.25:1/2);
end

function tb2_ouvrir2_Callback(hObject, eventdata, handles)
% hObject    handle to tb2_ouvrir2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[FileName,PathName] = uigetfile('*.mat','Choisir un signal (mat)');
if (~ischar(FileName) && FileName == 0) || (~ischar(PathName) && PathName == 0)
    return;
end

in=load([PathName FileName]);
str=fieldnames(in);
eval(['signal_filtre_tb2=in.' str{1} ';']);

if isstruct(signal_filtre_tb2)
    if ~isfield( signal_filtre_tb2, 'a' )
        display( 'Field "a" missing from structure' );
        return;
    end
    if ~isfield( signal_filtre_tb2, 'b' )
        display( 'Field "b" missing from structure' );
        return;
    end
    
    val = 0;
    cb=findobj('Tag','tb2_checkbox1');
    if ~isempty(cb)
        val=get(cb,'Value');
    end
    a = signal_filtre_tb2.a;
    b = signal_filtre_tb2.b;
    [rf ri] = DrawFilterInAxes( a, b, handles.tb2_filtre_f, '', handles.tb2_filtre_t, 'signal_filtre_tb2', val);
    
else
    axes(handles.tb2_filtre_t)
    h=stem(signal_filtre_tb2);
    set(h,'Tag','signal_filtre_tb2')
    xlabel('t');
    ylabel('x(t)');
    signal_fft=fftshift(fft(signal_filtre_tb2));
    axes(handles.tb2_filtre_f)
    xf=-0.5:1/length(signal_filtre_tb2):0.5;
    xf=xf(1:length(signal_filtre_tb2));
    cb=findobj('Tag','tb2_checkbox1');
    if ~isempty(cb)
        val=get(cb,'Value');
    else
        val=0;
    end
    if val==1
        plot(xf,20*log10(abs(signal_fft)));
    else
        plot(xf,abs(signal_fft));
    end
    xlabel('w/we');
    ylabel('|X(w/we)|');
    set(handles.tb2_filtre_f,'Xtick', -1/2:0.25:1/2);
    a = 1;
    b = signal_filtre_tb2;
end


% signal de sortie
h=findobj('Tag','signal_entree_tb2');
if ~isempty(h)
    signal_entree_tb2=get(h,'YData');
    signal_sortie=filter(b,a,signal_entree_tb2);
    axes(handles.tb2_sortie_t)
    h=plot((signal_sortie));%abs
    set(h,'Tag','signal_sortie_tb2')
    xlabel('t');
    ylabel('x(t)');
    signal_sortie_fft=fftshift(fft(signal_sortie));
    axes(handles.tb2_sortie_f)
    xf=-0.5:1/length(signal_sortie):0.5;
    xf=xf(1:length(signal_sortie));
    
    cb=findobj('Tag','tb2_checkbox1');
    if ~isempty(cb)
        val=get(cb,'Value');
    else
        val=0;
    end
    if val==1
        plot(xf,20*log10(abs(signal_sortie_fft)));
    else
        plot(xf,abs(signal_sortie_fft));
    end
     xlabel('w/we');
    ylabel('|X(w/we)|');
    set(handles.tb2_sortie_f,'Xtick', -1/2:0.25:1/2);
end

function tb2_checkbox1_Callback(hObject, eventdata, handles)
% hObject    handle to tb2_checkbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
cb=findobj('Tag','tb2_checkbox1');
if ~isempty(cb)
    val=get(cb,'Value');
    h=findobj('Tag','signal_entree_tb2');
    if ~isempty(h)
        signal_entree_tb2=get(h,'YData');
        signal_fft=fftshift(fft(signal_entree_tb2));
        axes(handles.tb2_entree_f)
        xf=-pi:2*pi/length(signal_entree_tb2):pi;
        xf=xf(1:length(signal_entree_tb2));
        
        if val==1
            plot(xf,20*log10(abs(signal_fft)));
        else
            plot(xf,abs(signal_fft));
        end
    end
    h=findobj('Tag','signal_filtre_tb2');
    if ~isempty(h)
        signal_filtre_tb2=get(h,'YData');
        signal_fft=fftshift(fft(signal_filtre_tb2));
        axes(handles.tb2_filtre_f)
        xf=-pi:2*pi/length(signal_filtre_tb2):pi;
        xf=xf(1:length(signal_filtre_tb2));
        if val==1
            plot(xf,20*log10(abs(signal_fft)));
        else
            plot(xf,abs(signal_fft));
        end
    end
    h=findobj('Tag','signal_sortie_tb2');
    if ~isempty(h)
        signal_sortie_tb2=get(h,'YData');
        signal_fft=fftshift(fft(signal_sortie_tb2));
        axes(handles.tb2_sortie_f)
        xf=-pi:2*pi/length(signal_sortie_tb2):pi;
        xf=xf(1:length(signal_sortie_tb2));
        if val==1
            plot(xf,20*log10(abs(signal_fft)));
        else
            plot(xf,abs(signal_fft));
        end
    end
end

function tb2_sauvegarder_sortie_Callback(hObject, eventdata, handles)
% hObject    handle to tb2_sauvegarder_sortie (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

h=findobj('Tag','signal_sortie_tb2');
if ~isempty(h)
    signal_postfiltre=get(h,'YData');
else
    signal_postfiltre=[];
end
[file,path] = uiputfile('*.mat','Sauvegarder le signal de corrï¿½ation');
if (~ischar(file) && file == 0) || (~ischar(path) && path == 0)
    return;
end
save([path file],'signal_postfiltre');

function tbshan_open_Callback(hObject, eventdata, handles)
% hObject    handle to tbshan_open (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

a=0.95;

signal_cx=[0:0.01:100];
signal_c=a.^(signal_cx);

slider_value_cos=get(handles.tbshan_slider2, 'Value');
slider_value_sin=get(handles.tbshan_slider3, 'Value');
signal_c= cos(2*pi*slider_value_cos*signal_cx)+sin(2*pi*slider_value_sin*signal_cx);
signal_cx=signal_cx(1:length(signal_c));

axes(handles.tbshan_out_t);
cla
h=plot(signal_cx,signal_c);
axis([0 50 -2 2])
xlabel('Temps t');
ylabel('x(t)');
slider_value=get(handles.tbshan_slider, 'Value');

Te=1/slider_value;
xte=signal_cx(1):Te:signal_cx(end);
signal_c_xte= cos(2*pi*slider_value_cos*xte) +  sin(2*pi*slider_value_sin*xte);
yte = interp1(signal_cx,signal_c,xte);
hold on, stem(xte,yte,'r');
% hold on, stem(xte,signal_c_xte,'g');

%graph frequentiels
Fe = 1/Te;
axes(handles.tbshan_out_f);
axis([-Fe/2 Fe/2 0 1.2])
xlabel('Fréquence f');
ylabel('Spectre |X(f)|');
axis square
cla



signal_fft=fft(signal_c_xte);
signal_fft=fft(signal_c_xte)/ numel( signal_c_xte );
FFT_T =  Fe*linspace( 0, 1, numel( signal_c_xte ))-Fe/2;

%axes(handles.tbshan_out_f);
% plot(x,y);
plot(FFT_T,fftshift(2*abs(signal_fft(1:numel(signal_c_xte)))));
% hold on, plot(x,y,'r--');
% hold on, plot(x-2*pi,y);
% hold on, plot(x+2*pi,y);
% hold on, plot(x-2*pi*slider_value,y,'r--');
% hold on, plot(x+2*pi*slider_value,y,'r--');

function tbshan_slider_Callback(hObject, eventdata, handles)
% hObject    handle to tbshan_slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
a=0.95;

signal_cx=[0:0.01:100];
signal_c=a.^(signal_cx);

slider_value_cos=get(handles.tbshan_slider2, 'Value');
slider_value_sin=get(handles.tbshan_slider3, 'Value');
signal_c= cos(2*pi*slider_value_cos*signal_cx)+sin(2*pi*slider_value_sin*signal_cx);
signal_cx=signal_cx(1:length(signal_c));

axes(handles.tbshan_out_t);
cla
h=plot(signal_cx,signal_c);
axis([0 50 -2 2])

if strcmp(get(hObject,'Type'),'uicontrol')
    slider_value = get(hObject,'Value');
    set(handles.tbshan_slidervalue,'String',slider_value);
else
    slider_value=1;
end

Te=1/slider_value;
xte=signal_cx(1):Te:signal_cx(end);
signal_c_xte= cos(2*pi*slider_value_cos*xte) +  sin(2*pi*slider_value_sin*xte);
yte = interp1(signal_cx,signal_c,xte);
hold on, stem(xte,yte,'r');
% hold on, stem(xte,signal_c_xte,'g');

%graph frequentiels
Fe = 1/Te;
axes(handles.tbshan_out_f);
axis([-Fe/2 Fe/2 0 1.2])
axis square
cla



signal_fft=fft(signal_c_xte);
signal_fft=fft(signal_c_xte)/ numel( signal_c_xte );
FFT_T =  Fe*linspace( 0, 1, numel( signal_c_xte ))-Fe/2;

%axes(handles.tbshan_out_f);
% plot(x,y);
plot(FFT_T,fftshift(2*abs(signal_fft(1:numel(signal_c_xte)))));
% hold on, plot(x,y,'r--');
% hold on, plot(x-2*pi,y);
% hold on, plot(x+2*pi,y);
% hold on, plot(x-2*pi*slider_value,y,'r--');
% hold on, plot(x+2*pi*slider_value,y,'r--');

function tbshan_slider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tbshan_slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

function tbshan_slidervalue_Callback(hObject, eventdata, handles)
% hObject    handle to tbshan_slidervalue (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

function tbshan_slidervalue_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tbshan_slidervalue (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function tbfir_win_Callback(hObject, eventdata, handles)
% hObject    handle to tbfir_win (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

function tbfir_sampling_Callback(hObject, eventdata, handles)
% hObject    handle to tbfir_sampling (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

function tbfir_optimization_Callback(hObject, eventdata, handles)
% hObject    handle to tbfir_optimization (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

function tbfir_compute_Callback(hObject, eventdata, handles)
% hObject    handle to tbfir_compute (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

function tbtfd_ouvrir1_Callback(hObject, eventdata, handles)
% hObject    handle to tb2_ouvrir1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



[FileName,PathName] = uigetfile('*.mat','Choisir un signal (mat)');
if (~ischar(FileName) && FileName == 0) || (~ischar(PathName) && PathName == 0)
    return;
end

in=load([PathName FileName]);
axes(handles.tbtfd_t)
str=fieldnames(in);
eval(['signal_entree_tbtfd=in.' str{1} ';']);
h=stem(signal_entree_tbtfd);
set(h,'Tag','signal_entree_tbtfd')
xlabel('t');
ylabel('x(t)');

set(handles.tbtfd_sigN,'String',length(signal_entree_tbtfd));
set(handles.tbtfd_inN,'String',length(signal_entree_tbtfd));

signal_fft=fftshift(fft(signal_entree_tbtfd));
axes(handles.tbtfd_f)

xf=-0.5:1/length(signal_entree_tbtfd):0.5;
xf=xf(1:length(signal_entree_tbtfd));
cb=findobj('Tag','tbtfd_cb');
if ~isempty(cb)
    val=get(cb,'Value');
else
    val=0;
end
if val==1
    plot(xf,20*log10(abs(signal_fft)));
else
    plot(xf,abs(signal_fft));
end
xlabel('w en frequence normalisée entre -1/2 et 1/2');
ylabel('|X(w)|');
set(handles.tbtfd_f,'Xtick', -1/2:0.1:1/2);


function tbtfd_sigN_Callback(hObject, eventdata, handles)
% hObject    handle to tbtfd_sigN (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA

function tbtfd_inN_Callback(hObject, eventdata, handles)
% hObject    handle to tbtfd_inN (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

function tbtfd_inN_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tbtfd_inN (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function tbtfd_calculer_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tbtfd_calculer (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

function tbtfd_calculer_Callback(hObject, eventdata, handles)
% hObject    handle to tbtfd_calculer (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
h=findobj('Tag','signal_entree_tbtfd');
if ~isempty(h)
    signal_entree_tbtfd=get(h,'YData');
    h=findobj('Tag','tbtfd_inN');
    sizeN=str2num(get(h,'String'));

    signal_fft=fftshift(fft(signal_entree_tbtfd,sizeN));
    axes(handles.tbtfd_f)
    cla;

    xf=-0.5:1/length(signal_fft):0.5;
    xf=xf(1:length(signal_fft));
    cb=findobj('Tag','tbtfd_cb');
    if ~isempty(cb)
        val=get(cb,'Value');
    else
        val=0;
    end
    if val==1
        plot(xf,20*log10(abs(signal_fft)));
    else
        plot(xf,abs(signal_fft));
    end
    
    xlabel('w en frequence normalisée entre -1/2 et 1/2');
    ylabel('|X(w)|');
    set(handles.tbtfd_f,'Xtick', -1/2:0.1:1/2);
end

function tbtfd_cb_Callback(hObject, eventdata, handles)
% hObject    handle to tbtfd_cb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of tbtfd_cb
cb=findobj('Tag','tbtfd_cb');
if ~isempty(cb)
    val=get(cb,'Value');
    h=findobj('Tag','signal_entree_tbtfd');
    if ~isempty(h)
        g=findobj('Tag','tbtfd_inN');
        if ~isempty(g)
            sizeN=str2num(get(g,'String'));
        else
            sizeN=length(signal_entree_tbtfd);
        end
        
        
        signal_entree_tbtfd=get(h,'YData');
        signal_fft=fftshift(fft(signal_entree_tbtfd,sizeN));
        axes(handles.tbtfd_f)
        xf=-pi:2*pi/length(signal_fft):pi;
        xf=xf(1:length(signal_fft));
        
        if val==1
            plot(xf,20*log10(abs(signal_fft)));
        else
            plot(xf,abs(signal_fft));
        end
    end
end


% --- Executes during object creation, after setting all properties.
function coeff_a_CreateFcn(hObject, eventdata, handles)
% hObject    handle to coeff_a (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function coeff_a_Callback(hObject, eventdata, handles)
% hObject    handle to coeff_a (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of coeff_a as text
%        str2double(get(hObject,'String')) returns contents of coeff_a as a double


% --- Executes during object creation, after setting all properties.
function coeff_b_CreateFcn(hObject, eventdata, handles)
% hObject    handle to coeff_b (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function coeff_b_Callback(hObject, eventdata, handles)
% hObject    handle to coeff_b (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of coeff_b as text
%        str2double(get(hObject,'String')) returns contents of coeff_b as a double





% --- Executes on button press in tb4_sauvegarder.
function tb4_ouvrir_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_sauvegarder (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
tb4_specifier_Callback(hObject, eventdata, handles);
set(findobj('Tag','tb4_specifier'),'Value',1);


[FileName,PathName] = uigetfile('*.mat','Choisir un signal (mat)');

in=load([PathName FileName]);
axes(handles.tb4_ri)
str=fieldnames(in);
eval(['tb4_ri=in.' str{1} ';']);


if isstruct(tb4_ri)
    a=tb4_ri.a;
    b=tb4_ri.b;

    axes(handles.tb4_poles);
    cla;
    zplane(a,b,handles.tb4_poles);

    rf=freqz(b,a,'whole');
    cb=findobj('Tag','tb4_dbcb');
    if ~isempty(cb)
        val=get(cb,'Value');
        axes(handles.tb4_rf)
        cla;
        signal_fft=abs(rf);
        xf=-0.5:1/length(signal_fft):0.5;
        xf=xf(1:length(signal_fft));

        if val==1
            plot(xf,20*log10(fftshift(signal_fft)));
        else
            plot(xf,fftshift(signal_fft));
        end

        xlabel('w/we');
        ylabel('|X(w/we)|');
        set(handles.tb4_rf,'Xtick', -1/2:0.25:1/2);
    end


    % reponse impulsionnelle
    imp=[1; zeros(49,1)];
    ri = filter(b,a,imp);
    axes(handles.tb4_ri);
    cla;
    h=stem(ri);
    set(h,'Tag','signal_tb4ri')

    set(findobj('Tag','tb4_ai'),'String',['[' num2str(a) ']']);
    set(findobj('Tag','tb4_bi'),'String',['[' num2str(b) ']']);
    
else
    if size(tb4_ri,2)<size(tb4_ri,1)
        tb4_ri=tb4_ri';
    end

    h=stem(tb4_ri);
    set(h,'Tag','signal_tb4ri')

    N=length(tb4_ri);
    while tb4_ri(N)<1E-4
        N=N-1;
    end
    set(findobj('Tag','tb4_Nri'),'String',num2str(N));


    set(findobj('Tag','tb4_bi'),'String',['[' num2str(tb4_ri) ']']);

    signal_fft=fftshift(fft(tb4_ri));
    axes(handles.tb4_rf)
    xf=-0.5:1/length(tb4_ri):0.5;
    xf=xf(1:length(tb4_ri));
    cb=findobj('Tag','tb4_dbcb');
    if ~isempty(cb)
        val=get(cb,'Value');
    else
        val=0;
    end
    if val==1
        plot(xf,20*log10(abs(signal_fft)));
    else
        plot(xf,abs(signal_fft));
    end
    xlabel('w/we');
    ylabel('|X(w/we)|');
    set(handles.tb4_rf,'Xtick', -1/2:0.25:1/2);
    axes(handles.tb4_poles);
    cla;
    zplane(tb4_ri,[1],handles.tb4_poles);
end


% --- Executes on button press in tb4_sauvegarder.
function tb4_sauvegarder_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_sauvegarder (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if get(findobj('Tag','tb4_radiobutton1'),'Value')==1

    eval(['a=' get(findobj('Tag','tb4_ai'),'String') ';']);
    eval(['b=' get(findobj('Tag','tb4_bi'),'String') ';']);
    
    aibi.a=a;
    aibi.b=b;
    
    [file,path] = uiputfile('*.mat','Sauvegarder les coefficients ai/bi');
    if (~ischar(file) && file == 0) || (~ischar(path) && path == 0)
        return;
    end
    save([path file],'aibi');
    
    
elseif get(findobj('Tag','tb4_radiobutton2'),'Value')==1
    
    g=findobj('Tag','tb4_Nri');
    if ~isempty(g)
        eval(['Nri=' get(g,'String') ';']);
    else
        Nri=50;
    end
    imp=[1; zeros(Nri,1)];
    ri = filter(coeff_bi,coeff_ai,imp);
    
    [file,path] = uiputfile('*.mat','Sauvegarder la reponse impulsionnelle');
    if (~ischar(file) && file == 0) || (~ischar(path) && path == 0)
        return;
    end
    save([path file],'ri');
    
end



% --- Executes on button press in tb4_radiobutton1.
function tb4_radiobutton1_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_radiobutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of tb4_radiobutton1
set(findobj('Tag','tb4_radiobutton2'),'Value',0);

% --- Executes on button press in tb4_radiobutton2.
function tb4_radiobutton2_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_radiobutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of tb4_radiobutton2
set(findobj('Tag','tb4_radiobutton1'),'Value',0);

h=findobj('Tag','signal_tb4ri');
if ~isempty(h)
    signal_tb4ri=get(h,'YData');
    N=length(signal_tb4ri);
    while signal_tb4ri(N)<1E-4
        N=N-1;
    end
    set(findobj('Tag','tb4_Nri'),'String',num2str(N));
end




function tb4_Nri_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_Nri (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb4_Nri as text
%        str2double(get(hObject,'String')) returns contents of tb4_Nri as a double


% --- Executes during object creation, after setting all properties.
function tb4_Nri_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb4_Nri (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function tb4_prelude(hObject, eventdata, handles)
axes(handles.tb4_poles)
cla;
axes(handles.tb4_ri)
cla;
axes(handles.tb4_rf)
cla;

set(findobj('Tag','tb4_text6'),'Visible','off');
set(findobj('Tag','tb4_radiobutton1'),'Visible','off');
set(findobj('Tag','tb4_radiobutton2'),'Visible','off');
set(findobj('Tag','tb4_Nri'),'Visible','off');
set(findobj('Tag','tb4_sauvegarder'),'Visible','off');


% --- Executes on button press in tb4_butter_rb.
function tb4_butter_rb_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_butter_rb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


tb4_prelude(hObject, eventdata, handles)



set(findobj('Tag','tb4_ellip_rb'),'Value',0);
set(findobj('Tag','tb4_cheby1_rb'),'Value',0);
set(findobj('Tag','tb4_cheby2_rb'),'Value',0);
set(findobj('Tag','tb4_specifier'),'Value',0);

set(findobj('Tag','tb4_textRs'),'Visible','off');
set(findobj('Tag','tb4_textRP'),'Visible','off');
set(findobj('Tag','tb4_Rp'),'Visible','off');
set(findobj('Tag','tb4_Rs'),'Visible','off');

set(findobj('Tag','tb4_low'),'Visible','on');
set(findobj('Tag','tb4_high'),'Visible','on');
set(findobj('Tag','tb4_stop'),'Visible','on');
set(findobj('Tag','tb4_band'),'Visible','on');

set(findobj('Tag','tb4_avec'),'Visible','on');
set(findobj('Tag','tb4_textordre'),'Visible','on');
set(findobj('Tag','tb4_ordre'),'Visible','on');
set(findobj('Tag','tb4_Wn'),'Visible','on');

set(findobj('Tag','tb4_text3'),'Visible','off');
set(findobj('Tag','tb4_text4'),'Visible','off');
set(findobj('Tag','tb4_ai'),'Visible','off');
set(findobj('Tag','tb4_bi'),'Visible','off');


% --- Executes on button press in tb4_ellip_rb.
function tb4_ellip_rb_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_ellip_rb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

tb4_prelude(hObject, eventdata, handles)



set(findobj('Tag','tb4_cheby1_rb'),'Value',0);
set(findobj('Tag','tb4_butter_rb'),'Value',0);
set(findobj('Tag','tb4_cheby2_rb'),'Value',0);
set(findobj('Tag','tb4_specifier'),'Value',0);

set(findobj('Tag','tb4_textRs'),'Visible','on');
set(findobj('Tag','tb4_textRP'),'Visible','on');
set(findobj('Tag','tb4_Rp'),'Visible','on');
set(findobj('Tag','tb4_Rs'),'Visible','on');

set(findobj('Tag','tb4_low'),'Visible','on');
set(findobj('Tag','tb4_high'),'Visible','on');
set(findobj('Tag','tb4_stop'),'Visible','on');
set(findobj('Tag','tb4_band'),'Visible','on');

set(findobj('Tag','tb4_avec'),'Visible','on');
set(findobj('Tag','tb4_textordre'),'Visible','on');
set(findobj('Tag','tb4_ordre'),'Visible','on');
set(findobj('Tag','tb4_Wn'),'Visible','on');

set(findobj('Tag','tb4_text3'),'Visible','off');
set(findobj('Tag','tb4_text4'),'Visible','off');
set(findobj('Tag','tb4_ai'),'Visible','off');
set(findobj('Tag','tb4_bi'),'Visible','off');



% --- Executes on button press in tb4_cheby1_rb.
function tb4_cheby1_rb_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_cheby1_rb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

tb4_prelude(hObject, eventdata, handles)


set(findobj('Tag','tb4_ellip_rb'),'Value',0);
set(findobj('Tag','tb4_butter_rb'),'Value',0);
set(findobj('Tag','tb4_cheby2_rb'),'Value',0);
set(findobj('Tag','tb4_specifier'),'Value',0);

set(findobj('Tag','tb4_textRs'),'Visible','off');
set(findobj('Tag','tb4_textRP'),'Visible','on');
set(findobj('Tag','tb4_Rp'),'Visible','on');
set(findobj('Tag','tb4_Rs'),'Visible','off');

set(findobj('Tag','tb4_low'),'Visible','on');
set(findobj('Tag','tb4_high'),'Visible','on');
set(findobj('Tag','tb4_stop'),'Visible','on');
set(findobj('Tag','tb4_band'),'Visible','on');

set(findobj('Tag','tb4_avec'),'Visible','on');
set(findobj('Tag','tb4_textordre'),'Visible','on');
set(findobj('Tag','tb4_ordre'),'Visible','on');
set(findobj('Tag','tb4_Wn'),'Visible','on');

set(findobj('Tag','tb4_text3'),'Visible','off');
set(findobj('Tag','tb4_text4'),'Visible','off');
set(findobj('Tag','tb4_ai'),'Visible','off');
set(findobj('Tag','tb4_bi'),'Visible','off');


% --- Executes on button press in tb4_cheby2_rb.
function tb4_cheby2_rb_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_cheby2_rb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

tb4_prelude(hObject, eventdata, handles)


set(findobj('Tag','tb4_ellip_rb'),'Value',0);
set(findobj('Tag','tb4_cheby1_rb'),'Value',0);
set(findobj('Tag','tb4_butter_rb'),'Value',0);
set(findobj('Tag','tb4_specifier'),'Value',0);

set(findobj('Tag','tb4_textRs'),'Visible','on');
set(findobj('Tag','tb4_textRP'),'Visible','off');
set(findobj('Tag','tb4_Rp'),'Visible','off');
set(findobj('Tag','tb4_Rs'),'Visible','on');

set(findobj('Tag','tb4_low'),'Visible','on');
set(findobj('Tag','tb4_high'),'Visible','on');
set(findobj('Tag','tb4_stop'),'Visible','on');
set(findobj('Tag','tb4_band'),'Visible','on');

set(findobj('Tag','tb4_avec'),'Visible','on');
set(findobj('Tag','tb4_textordre'),'Visible','on');
set(findobj('Tag','tb4_ordre'),'Visible','on');
set(findobj('Tag','tb4_Wn'),'Visible','on');

set(findobj('Tag','tb4_text3'),'Visible','off');
set(findobj('Tag','tb4_text4'),'Visible','off');
set(findobj('Tag','tb4_ai'),'Visible','off');
set(findobj('Tag','tb4_bi'),'Visible','off');


% --- Executes on button press in tb4_cheby2_rb.
function tb4_specifier_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_cheby2_rb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

tb4_prelude(hObject, eventdata, handles)


set(findobj('Tag','tb4_ellip_rb'),'Value',0);
set(findobj('Tag','tb4_cheby1_rb'),'Value',0);
set(findobj('Tag','tb4_butter_rb'),'Value',0);
set(findobj('Tag','tb4_cheby2_rb'),'Value',0);

set(findobj('Tag','tb4_textRs'),'Visible','off');
set(findobj('Tag','tb4_textRP'),'Visible','off');
set(findobj('Tag','tb4_Rp'),'Visible','off');
set(findobj('Tag','tb4_Rs'),'Visible','off');

set(findobj('Tag','tb4_low'),'Visible','off');
set(findobj('Tag','tb4_high'),'Visible','off');
set(findobj('Tag','tb4_stop'),'Visible','off');
set(findobj('Tag','tb4_band'),'Visible','off');

set(findobj('Tag','tb4_avec'),'Visible','off');
set(findobj('Tag','tb4_textordre'),'Visible','off');
set(findobj('Tag','tb4_ordre'),'Visible','off');
set(findobj('Tag','tb4_Wn'),'Visible','off');
set(findobj('Tag','tb4_Wn2'),'Visible','off');

set(findobj('Tag','tb4_text3'),'Visible','on');
set(findobj('Tag','tb4_text4'),'Visible','on');
set(findobj('Tag','tb4_ai'),'Visible','on');
set(findobj('Tag','tb4_bi'),'Visible','on');

set(findobj('Tag','tb4_ai'),'String','[]');
set(findobj('Tag','tb4_bi'),'String','[]');





function tb4_Wn_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_Wn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb4_Wn as text
%        str2double(get(hObject,'String')) returns contents of tb4_Wn as a double


% --- Executes during object creation, after setting all properties.
function tb4_Wn_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb4_Wn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in tb4_generate_coeff.
function tb4_generate_coeff_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_generate_coeff (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(findobj('Tag','tb4_text6'),'Visible','on');
set(findobj('Tag','tb4_radiobutton1'),'Visible','on');
set(findobj('Tag','tb4_radiobutton2'),'Visible','on');
set(findobj('Tag','tb4_Nri'),'Visible','on');
set(findobj('Tag','tb4_sauvegarder'),'Visible','on');

h_wn=findobj('Tag','tb4_Wn');
h_ordre=findobj('Tag','tb4_ordre');
if ~isempty(h_wn) && ~isempty(h_ordre)
    eval(['Wn=' get(h_wn,'String') ';']);
    eval(['ordre=' get(h_ordre,'String') ';']);
end


type='low';
if get(findobj('Tag','tb4_stop'),'Value')
    type='stop';
    h_wn2=findobj('Tag','tb4_Wn2');
    if ~isempty(h_wn)
        eval(['Wn=[Wn ' get(h_wn2,'String') '];']);
    end
elseif get(findobj('Tag','tb4_band'),'Value')
    type='bandpass';
    h_wn2=findobj('Tag','tb4_Wn2');
    if ~isempty(h_wn)
        eval(['Wn=[Wn ' get(h_wn2,'String') '];']);
    end
elseif get(findobj('Tag','tb4_low'),'Value')
    type='low';
elseif get(findobj('Tag','tb4_high'),'Value')
    type='high';
end

h_ai=findobj('Tag','tb4_ai');
h_bi=findobj('Tag','tb4_bi');

h_rp=findobj('Tag','tb4_Rp');
h_rs=findobj('Tag','tb4_Rs');

if ~isempty(h_rp)
    eval(['Rp=' get(h_rp,'String') ';']);
end
if ~isempty(h_rs)
    eval(['Rs=' get(h_rs,'String') ';']);
end

if get(findobj('Tag','tb4_butter_rb'),'Value')
    [b,a]=butter(ordre,Wn*2,type);
elseif get(findobj('Tag','tb4_ellip_rb'),'Value')
    [b,a]=ellip(ordre,Rp,Rs,Wn*2,type);
elseif get(findobj('Tag','tb4_cheby1_rb'),'Value')
    [b,a]=cheby1(ordre,Rp,Wn*2,type);
elseif get(findobj('Tag','tb4_cheby2_rb'),'Value')
    [b,a]=cheby2(ordre,Rs,Wn*2,type);
elseif get(findobj('Tag','tb4_specifier'),'Value')
    eval(['a=' get(findobj('Tag','tb4_ai'),'String') ';']);
    eval(['b=' get(findobj('Tag','tb4_bi'),'String') ';']);
end

if ~isempty(a)
    set(h_ai,'String',['[' num2str(a) ']']);
end
if ~isempty(b)
    set(h_bi,'String',['[' num2str(b) ']']);
end

set(findobj('Tag','tb4_text3'),'Visible','on');
set(findobj('Tag','tb4_text4'),'Visible','on');
set(findobj('Tag','tb4_ai'),'Visible','on');
set(findobj('Tag','tb4_bi'),'Visible','on');

% global coeff_ai;
% global coeff_bi;

axes(handles.tb4_poles);
cla;
eval(['a=' get(handles.tb4_ai,'String') ';']);
eval(['b=' get(handles.tb4_bi,'String') ';']);
% coeff_ai=a;
% coeff_bi=b;

zplane(b,a,handles.tb4_poles);

cb=findobj('Tag','tb4_dbcb');
val = 0;
if ~isempty(cb)
    val=get(cb,'Value');
end
[rf ri] = DrawFilterInAxes( a, b, handles.tb4_rf, 'UNUSED', handles.tb4_ri, 'signal_tb4ri', val );


function tb4_ordre_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_ordre (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

function tb4_ordre_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb4_ordre (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function tb4_Rs_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_Rs (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

function tb4_Rp_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_Rp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

function tb4_textRP_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb4_textRP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

function tb4_textRs_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb4_textRs (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

function tb4_high_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_high (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(findobj('Tag','tb4_low'),'Value',0);
set(findobj('Tag','tb4_band'),'Value',0);
set(findobj('Tag','tb4_stop'),'Value',0);
set(findobj('Tag','tb4_Wn2'),'Visible','off');

function tb4_low_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_low (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(findobj('Tag','tb4_band'),'Value',0);
set(findobj('Tag','tb4_high'),'Value',0);
set(findobj('Tag','tb4_stop'),'Value',0);
set(findobj('Tag','tb4_Wn2'),'Visible','off');

function tb4_band_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_band (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(findobj('Tag','tb4_low'),'Value',0);
set(findobj('Tag','tb4_high'),'Value',0);
set(findobj('Tag','tb4_stop'),'Value',0);
set(findobj('Tag','tb4_Wn2'),'Visible','on');

function tb4_stop_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_stop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(findobj('Tag','tb4_low'),'Value',0);
set(findobj('Tag','tb4_band'),'Value',0);
set(findobj('Tag','tb4_high'),'Value',0);
set(findobj('Tag','tb4_Wn2'),'Visible','on');

function tb4_dbcb_Callback(hObject, eventdata, handles)
% hObject    handle to tbtfd_cb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of tbtfd_cb
cb=findobj('Tag','tb4_dbcb');
if ~isempty(cb)
    val=get(cb,'Value');
    h=findobj('Tag','signal_tb4ri');
    if ~isempty(h)
        signal_tb4ri=get(h,'YData');
        sizeN=length(signal_tb4ri);
        
        signal_fft=fftshift(fft(signal_tb4ri,sizeN));
        axes(handles.tb4_rf)
        xf=-0.5:1/length(signal_fft):0.5;
        xf=xf(1:length(signal_fft));
        
        if val==1
            plot(xf,20*log10(abs(signal_fft)));
        else
            plot(xf,abs(signal_fft));
        end
        xlabel('w/we');
        ylabel('|X(w/we)|');
        set(handles.tb4_rf,'Xtick', -1/2:0.25:1/2);
    end
end



function tb3_ouvrir_Callback(hObject, eventdata, handles)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

tb3_specifier_Callback(hObject, eventdata, handles);
set(findobj('Tag','tb3_specifier'),'Value',1);


[FileName,PathName] = uigetfile('*.mat','Choisir un signal (mat)');
if (~ischar(FileName) && FileName == 0) || (~ischar(PathName) && PathName == 0)
    return;
end

in=load([PathName FileName]);
axes(handles.tb3_ri)
str=fieldnames(in);
eval(['tb3_ri=in.' str{1} ';']);

if size(tb3_ri,2)<size(tb3_ri,1)
    tb3_ri=tb3_ri';
end

h=stem(tb3_ri);
set(h,'Tag','signal_tb3ri')

N=length(tb3_ri);
while tb3_ri(N)<1E-4
    N=N-1;
end
set(findobj('Tag','tb3_Nri'),'String',num2str(N));


set(findobj('Tag','tb3_bi'),'String',['[' num2str(tb3_ri) ']']);

signal_fft=fftshift(fft(tb3_ri));
axes(handles.tb3_rf)
xf=-0.5:1/length(tb3_ri):0.5;
xf=xf(1:length(tb3_ri));
cb=findobj('Tag','tb3_dbcb');
if ~isempty(cb)
    val=get(cb,'Value');
else
    val=0;
end
if val==1
    plot(xf,20*log10(abs(signal_fft)));
else
    plot(xf,abs(signal_fft));
end
xlabel('w/we');
ylabel('|X(w/we)|');
set(handles.tb3_rf,'Xtick', -1/2:0.25:1/2);
axes(handles.tb3_poles);
cla;

zplane(tb3_ri,[1],handles.tb3_poles);




% --- Executes on button press in tb3_sauvegarder.
function tb3_sauvegarder_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_sauvegarder (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

eval(['bi=' get(findobj('Tag','tb3_bi'),'String') ';']);

g=findobj('Tag','tb3_Nri');
if ~isempty(g)
    eval(['Nri=' get(g,'String') ';']);
else
    Nri=50;
end
ri = [bi zeros(1,Nri-length(bi))];

[file,path] = uiputfile('*.mat','Sauvegarder la rï¿½onse impulsionnelle');
if (~ischar(file) && file == 0) || (~ischar(path) && path == 0)
    return;
end
save([path file],'ri');





function tb3_Nri_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_Nri (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb3_Nri as text
%        str2double(get(hObject,'String')) returns contents of tb3_Nri as a double


% --- Executes during object creation, after setting all properties.
function tb3_Nri_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb3_Nri (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function tb3_prelude(hObject, eventdata, handles)
axes(handles.tb3_poles)
cla;
axes(handles.tb3_ri)
cla;
axes(handles.tb3_rf)
cla;

set(findobj('Tag','tb3_textsauvegarder'),'Visible','off');
set(findobj('Tag','tb3_Nri'),'Visible','off');
set(findobj('Tag','tb3_sauvegarder'),'Visible','off');



% --- Executes on button press in tb3_butter_rb.
function tb3_fir1_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_butter_rb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

tb3_prelude(hObject, eventdata, handles)

set(findobj('Tag','tb3_fir2'),'Value',0);
set(findobj('Tag','tb3_optimisation'),'Value',0);
set(findobj('Tag','tb3_specifier'),'Value',0);



set(findobj('Tag','tb3_textf'),'Visible','off');
set(findobj('Tag','tb3_textm'),'Visible','off');
set(findobj('Tag','tb3_f'),'Visible','off');
set(findobj('Tag','tb3_m'),'Visible','off');

set(findobj('Tag','tb3_wn'),'Visible','on');
set(findobj('Tag','tb3_wc1'),'Visible','on');
set(findobj('Tag','tb3_wc2'),'Visible','on');

set(findobj('Tag','tb3_textordre'),'Visible','on');
set(findobj('Tag','tb3_ordre'),'Visible','on');

set(findobj('Tag','tb3_bi'),'Visible','off');
set(findobj('Tag','tb3_textb2'),'Visible','off');

set(findobj('Tag','tb3_low'),'Visible','on');
set(findobj('Tag','tb3_band'),'Visible','on');
set(findobj('Tag','tb3_stop'),'Visible','on');
set(findobj('Tag','tb3_high'),'Visible','on');

set(findobj('Tag','tb3_win1'),'Visible','on');
set(findobj('Tag','tb3_win2'),'Visible','on');
set(findobj('Tag','tb3_win3'),'Visible','on');
set(findobj('Tag','tb3_win4'),'Visible','on');
set(findobj('Tag','tb3_win5'),'Visible','on');
set(findobj('Tag','tb3_win6'),'Visible','on');


% --- Executes on button press in tb3_ellip_rb.
function tb3_fir2_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_ellip_rb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
tb3_prelude(hObject, eventdata, handles)

set(findobj('Tag','tb3_fir1'),'Value',0);
set(findobj('Tag','tb3_optimisation'),'Value',0);
set(findobj('Tag','tb3_specifier'),'Value',0);

set(findobj('Tag','tb3_wn'),'Visible','off');
set(findobj('Tag','tb3_wc1'),'Visible','off');
set(findobj('Tag','tb3_wc2'),'Visible','off');

set(findobj('Tag','tb3_textf'),'Visible','on');
set(findobj('Tag','tb3_textm'),'Visible','on');
set(findobj('Tag','tb3_f'),'Visible','on');
set(findobj('Tag','tb3_m'),'Visible','on');

set(findobj('Tag','tb3_textordre'),'Visible','on');
set(findobj('Tag','tb3_ordre'),'Visible','on');

set(findobj('Tag','tb3_bi'),'Visible','off');
set(findobj('Tag','tb3_textb2'),'Visible','off');

set(findobj('Tag','tb3_low'),'Visible','off');
set(findobj('Tag','tb3_band'),'Visible','off');
set(findobj('Tag','tb3_stop'),'Visible','off');
set(findobj('Tag','tb3_high'),'Visible','off');

set(findobj('Tag','tb3_win1'),'Visible','on');
set(findobj('Tag','tb3_win2'),'Visible','on');
set(findobj('Tag','tb3_win3'),'Visible','on');
set(findobj('Tag','tb3_win4'),'Visible','on');
set(findobj('Tag','tb3_win5'),'Visible','on');
set(findobj('Tag','tb3_win6'),'Visible','on');



% --- Executes on button press in tb3_cheby1_rb.
function tb3_optimisation_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_cheby1_rb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
tb3_prelude(hObject, eventdata, handles)

set(findobj('Tag','tb3_fir1'),'Value',0);
set(findobj('Tag','tb3_fir2'),'Value',0);
set(findobj('Tag','tb3_specifier'),'Value',0);

set(findobj('Tag','tb3_wn'),'Visible','off');
set(findobj('Tag','tb3_wc1'),'Visible','off');
set(findobj('Tag','tb3_wc2'),'Visible','off');

set(findobj('Tag','tb3_textf'),'Visible','on');
set(findobj('Tag','tb3_textm'),'Visible','on');
set(findobj('Tag','tb3_f'),'Visible','on');
set(findobj('Tag','tb3_m'),'Visible','on');

set(findobj('Tag','tb3_textordre'),'Visible','on');
set(findobj('Tag','tb3_ordre'),'Visible','on');

set(findobj('Tag','tb3_bi'),'Visible','off');
set(findobj('Tag','tb3_textb2'),'Visible','off');

set(findobj('Tag','tb3_low'),'Visible','off');
set(findobj('Tag','tb3_band'),'Visible','off');
set(findobj('Tag','tb3_stop'),'Visible','off');
set(findobj('Tag','tb3_high'),'Visible','off');

set(findobj('Tag','tb3_win1'),'Visible','off');
set(findobj('Tag','tb3_win2'),'Visible','off');
set(findobj('Tag','tb3_win3'),'Visible','off');
set(findobj('Tag','tb3_win4'),'Visible','off');
set(findobj('Tag','tb3_win5'),'Visible','off');
set(findobj('Tag','tb3_win6'),'Visible','off');



% --- Executes on button press in tb3_cheby2_rb.
function tb3_specifier_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_cheby2_rb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA
tb3_prelude(hObject, eventdata, handles)

set(findobj('Tag','tb3_fir1'),'Value',0);
set(findobj('Tag','tb3_fir2'),'Value',0);
set(findobj('Tag','tb3_optimisation'),'Value',0);

set(findobj('Tag','tb3_wn'),'Visible','off');
set(findobj('Tag','tb3_wc1'),'Visible','off');
set(findobj('Tag','tb3_wc2'),'Visible','off');

set(findobj('Tag','tb3_f'),'Visible','off');
set(findobj('Tag','tb3_m'),'Visible','off');
set(findobj('Tag','tb3_textf'),'Visible','off');
set(findobj('Tag','tb3_textm'),'Visible','off');

set(findobj('Tag','tb3_textordre'),'Visible','off');
set(findobj('Tag','tb3_ordre'),'Visible','off');

set(findobj('Tag','tb3_bi'),'Visible','on');
set(findobj('Tag','tb3_bi'),'String','[]');
set(findobj('Tag','tb3_textb2'),'Visible','on');

set(findobj('Tag','tb3_low'),'Visible','off');
set(findobj('Tag','tb3_band'),'Visible','off');
set(findobj('Tag','tb3_stop'),'Visible','off');
set(findobj('Tag','tb3_high'),'Visible','off');

set(findobj('Tag','tb3_win1'),'Visible','off');
set(findobj('Tag','tb3_win2'),'Visible','off');
set(findobj('Tag','tb3_win3'),'Visible','off');
set(findobj('Tag','tb3_win4'),'Visible','off');
set(findobj('Tag','tb3_win5'),'Visible','off');
set(findobj('Tag','tb3_win6'),'Visible','off');




function tb3_f_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_Wn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb3_Wn as text
%        str2double(get(hObject,'String')) returns contents of tb3_Wn as a double


% --- Executes during object creation, after setting all properties.
function tb3_f_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb3_Wn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function tb3_m_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_Wn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb3_Wn as text
%        str2double(get(hObject,'String')) returns contents of tb3_Wn as a double


% --- Executes during object creation, after setting all properties.
function tb3_m_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb3_Wn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function tb3_wc1_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_Wn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb3_Wn as text
%        str2double(get(hObject,'String')) returns contents of tb3_Wn as a double


% --- Executes during object creation, after setting all properties.
function tb3_wc1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb3_Wn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function tb3_wc2_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_Wn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb3_Wn as text
%        str2double(get(hObject,'String')) returns contents of tb3_Wn as a double


% --- Executes during object creation, after setting all properties.
function tb3_wc2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb3_Wn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function tb3_bi_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_Wn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb3_Wn as text
%        str2double(get(hObject,'String')) returns contents of tb3_Wn as a double


% --- Executes during object creation, after setting all properties.
function tb3_bi_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb3_Wn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in tb3_generate_coeff.
function tb3_generer_coeff_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_generate_coeff (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(findobj('Tag','tb3_textsauvegarder'),'Visible','on');
set(findobj('Tag','tb3_Nri'),'Visible','on');
set(findobj('Tag','tb3_sauvegarder'),'Visible','on');

a=[1];

h_wn=findobj('Tag','tb3_wc1');
h_ordre=findobj('Tag','tb3_ordre');
if ~isempty(h_wn) && ~isempty(h_ordre)
    eval(['Wn=' get(h_wn,'String') ';']);
    eval(['ordre=' get(h_ordre,'String') ';']);
end

% type
type='low';
if get(findobj('Tag','tb3_stop'),'Value')
    type='stop';
    h_wc2=findobj('Tag','tb3_wc2');
    if ~isempty(h_wc2)
        eval(['Wn=[Wn ' get(h_wc2,'String') '];']);
    end
elseif get(findobj('Tag','tb3_band'),'Value')
    type='bandpass';
    h_wc2=findobj('Tag','tb3_wc2');
    if ~isempty(h_wc2)
        eval(['Wn=[Wn ' get(h_wc2,'String') '];']);
    end
elseif get(findobj('Tag','tb3_low'),'Value')
    type='low';
elseif get(findobj('Tag','tb3_high'),'Value')
    type='high';
end

% fenetre
if get(findobj('Tag','tb3_win1'),'Value')
    fenetre='blackman(N)';
elseif get(findobj('Tag','tb3_win2'),'Value')
    fenetre='bartlett(N)';
elseif get(findobj('Tag','tb3_win3'),'Value')
    fenetre='hamming(N)';
elseif get(findobj('Tag','tb3_win4'),'Value')
    fenetre='hann(N)';
elseif get(findobj('Tag','tb3_win5'),'Value')
    fenetre='tukeywin(N)';
else
    fenetre='rectwin(N)';
end

h_bi=findobj('Tag','tb3_bi');

h_f=findobj('Tag','tb3_f');
h_m=findobj('Tag','tb3_m');

if ~isempty(h_f)
    eval(['f=' get(h_f,'String') ';']);
end
if ~isempty(h_m)
    eval(['m=' get(h_m,'String') ';']);
end

if get(findobj('Tag','tb3_fir1'),'Value')
    N=ordre+1;
    eval(['[b]=fir1(ordre,Wn,type,' fenetre ');']);
    set(findobj('Tag','tb3_bi'),'String',['[' num2str(b) ']']);
    set(findobj('Tag','tb3_bi'),'Visible','on');
    set(findobj('Tag','tb3_textb2'),'Visible','on');
elseif get(findobj('Tag','tb3_fir2'),'Value')
    N=ordre+1;
    
    eval(['f=' get(findobj('Tag','tb3_f'),'String') ';']);
    eval(['m=' get(findobj('Tag','tb3_m'),'String') ';']);
    eval(['[b]=fir2(ordre,f*2,m,' fenetre ')']);
    
    set(findobj('Tag','tb3_bi'),'String',['[' num2str(b) ']']);
    set(findobj('Tag','tb3_bi'),'Visible','on');
    set(findobj('Tag','tb3_textb2'),'Visible','on');
    
elseif get(findobj('Tag','tb3_optimisation'),'Value')
    N=ordre+1;
    
    eval(['f=' get(findobj('Tag','tb3_f'),'String') ';']);
    eval(['m=' get(findobj('Tag','tb3_m'),'String') ';']);
    eval(['[b]=firls(ordre,f*2,m)']);
    
    set(findobj('Tag','tb3_bi'),'String',['[' num2str(b) ']']);
    set(findobj('Tag','tb3_bi'),'Visible','on');
    set(findobj('Tag','tb3_textb2'),'Visible','on');
elseif get(findobj('Tag','tb3_specifier'),'Value')
    eval(['b=' get(findobj('Tag','tb3_bi'),'String') ';']);
end

if ~isempty(b)
    set(h_bi,'String',['[' num2str(b) ']']);
end


axes(handles.tb3_poles);
cla;

zplane(b,a,handles.tb3_poles);

cb=findobj('Tag','tb3_dbcb');
val = 0;
if ~isempty(cb)
    val=get(cb,'Value');
end
[rf ri] = DrawFilterInAxes( a, b, handles.tb3_rf, 'UNUSED', handles.tb3_ri, 'signal_tb3ri', val );

N=length(ri);
while ri(N)<1E-4
    N=N-1;
end
set(findobj('Tag','tb3_Nri'),'String',num2str(N));








function tb3_ordre_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_ordre (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

function tb3_ordre_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb3_ordre (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function tb3_high_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_high (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(findobj('Tag','tb3_low'),'Value',0);
set(findobj('Tag','tb3_band'),'Value',0);
set(findobj('Tag','tb3_stop'),'Value',0);
set(findobj('Tag','tb3_wc2'),'Visible','off');

function tb3_low_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_low (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(findobj('Tag','tb3_band'),'Value',0);
set(findobj('Tag','tb3_high'),'Value',0);
set(findobj('Tag','tb3_stop'),'Value',0);
set(findobj('Tag','tb3_wc2'),'Visible','off');

function tb3_band_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_band (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(findobj('Tag','tb3_low'),'Value',0);
set(findobj('Tag','tb3_high'),'Value',0);
set(findobj('Tag','tb3_stop'),'Value',0);
set(findobj('Tag','tb3_wc2'),'Visible','on');

function tb3_stop_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_stop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(findobj('Tag','tb3_low'),'Value',0);
set(findobj('Tag','tb3_band'),'Value',0);
set(findobj('Tag','tb3_high'),'Value',0);
set(findobj('Tag','tb3_wc2'),'Visible','on');

function tb3_dbcb_Callback(hObject, eventdata, handles)
% hObject    handle to tbtfd_cb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of tbtfd_cb
cb=findobj('Tag','tb3_dbcb');
if ~isempty(cb)
    val=get(cb,'Value');
    h=findobj('Tag','signal_tb3ri');
    if ~isempty(h)
        signal_tb3ri=get(h,'YData');
        sizeN=length(signal_tb3ri);
        
        signal_fft=fftshift(fft(signal_tb3ri,sizeN));
        axes(handles.tb3_rf)
        xf=-0.5:1/length(signal_fft):0.5;
        xf=xf(1:length(signal_fft));
        
        if val==1
            plot(xf,20*log10(abs(signal_fft)));
        else
            plot(xf,abs(signal_fft));
        end
        xlabel('w/we');
        ylabel('|X(w/we)|');
        set(handles.tb3_rf,'Xtick', -1/2:0.25:1/2);
    end
end


% --- Executes on button press in tb3_win1.
function tb3_win1_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_win1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(findobj('Tag','tb3_win2'),'Value',0);
set(findobj('Tag','tb3_win3'),'Value',0);
set(findobj('Tag','tb3_win4'),'Value',0);
set(findobj('Tag','tb3_win5'),'Value',0);
set(findobj('Tag','tb3_win6'),'Value',0);


% --- Executes on button press in tb3_win2.
function tb3_win2_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_win2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(findobj('Tag','tb3_win1'),'Value',0);
set(findobj('Tag','tb3_win3'),'Value',0);
set(findobj('Tag','tb3_win4'),'Value',0);
set(findobj('Tag','tb3_win5'),'Value',0);
set(findobj('Tag','tb3_win6'),'Value',0);


% --- Executes on button press in tb3_win3.
function tb3_win3_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_win3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(findobj('Tag','tb3_win1'),'Value',0);
set(findobj('Tag','tb3_win2'),'Value',0);
set(findobj('Tag','tb3_win4'),'Value',0);
set(findobj('Tag','tb3_win5'),'Value',0);
set(findobj('Tag','tb3_win6'),'Value',0);


% --- Executes on button press in tb3_win4.
function tb3_win4_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_win4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(findobj('Tag','tb3_win1'),'Value',0);
set(findobj('Tag','tb3_win2'),'Value',0);
set(findobj('Tag','tb3_win3'),'Value',0);
set(findobj('Tag','tb3_win5'),'Value',0);
set(findobj('Tag','tb3_win6'),'Value',0);


% --- Executes on button press in tb3_win5.
function tb3_win5_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_win5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(findobj('Tag','tb3_win1'),'Value',0);
set(findobj('Tag','tb3_win2'),'Value',0);
set(findobj('Tag','tb3_win3'),'Value',0);
set(findobj('Tag','tb3_win4'),'Value',0);
set(findobj('Tag','tb3_win6'),'Value',0);


% --- Executes on button press in tb3_win6.
function tb3_win6_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_win6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(findobj('Tag','tb3_win1'),'Value',0);
set(findobj('Tag','tb3_win2'),'Value',0);
set(findobj('Tag','tb3_win3'),'Value',0);
set(findobj('Tag','tb3_win4'),'Value',0);
set(findobj('Tag','tb3_win5'),'Value',0);

function [RF RI] = DrawFilterInAxes( CoeffA, CoeffB, HandleAxesRF, TagAxesRF, HandleAxesRI, TagAxesRI, IsDecibelOn )
    RF=freqz(CoeffB,CoeffA,'whole');
    
    axes(HandleAxesRF)
    cla;
    signal_fft=abs(RF);
    xf=-0.5:1/length(signal_fft):0.5;
    xf=xf(1:length(signal_fft));

    if IsDecibelOn==1
        plot(xf,20*log10(fftshift(signal_fft)));
    else
        plot(xf,fftshift(signal_fft));
    end
    xlabel('w/we');
    ylabel('|X(w/we)|');
    set(HandleAxesRF,'Xtick', -1/2:0.25:1/2);

    % reponse impulsionnelle
    imp=[1; zeros(49,1)];
    RI = filter(CoeffB,CoeffA,imp);
    axes(HandleAxesRI);
    cla;
    h=stem(RI);
    set(h,'Tag',TagAxesRI)
    xlabel('t');
    ylabel('x(t)');


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over buttoncorrel.
function buttoncorrel_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to buttoncorrel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
gcbo=hObject;
TP_TEMP_VAR.H = get(get(gcbo,'Parent'),'UserData');
TP_TEMP_VAR.old = get(TP_TEMP_VAR.H.TopPanel,'UserData');
if TP_TEMP_VAR.H.Tab(TP_TEMP_VAR.old)~=gcbo;   
    TP_TEMP_VAR.h = [TP_TEMP_VAR.H.Main TP_TEMP_VAR.H.Panel TP_TEMP_VAR.H.TopPanel TP_TEMP_VAR.H.Tab TP_TEMP_VAR.H.TopTab];
    TP_TEMP_VAR.units = get(TP_TEMP_VAR.h,'units');
    set(TP_TEMP_VAR.h,'units','pixel');
    TP_TEMP_VAR.p = get(gcbo,'pos');
    TP_TEMP_VAR.pTP = get(TP_TEMP_VAR.H.TopPanel,'pos');
    set(TP_TEMP_VAR.H.Tab(TP_TEMP_VAR.old),'pos',[TP_TEMP_VAR.pTP(1) TP_TEMP_VAR.p(2) TP_TEMP_VAR.pTP(3) TP_TEMP_VAR.p(4)],'back',TP_TEMP_VAR.H.Color{3},'fore',TP_TEMP_VAR.H.Color{4},'enable',TP_TEMP_VAR.H.Enable);
    TP_TEMP_VAR.pP = get(TP_TEMP_VAR.H.Panel(1),'pos');
    set(TP_TEMP_VAR.H.TopPanel,'pos',[TP_TEMP_VAR.p(1) TP_TEMP_VAR.pP(4) TP_TEMP_VAR.p(3) TP_TEMP_VAR.pTP(4)],'UserData',find(TP_TEMP_VAR.H.Tab==gcbo));
    set(TP_TEMP_VAR.H.TopTab  ,'pos',[TP_TEMP_VAR.p(1)+1 TP_TEMP_VAR.pP(4) TP_TEMP_VAR.p(3)-2 1]);
    set(gcbo,'pos',TP_TEMP_VAR.p+[1 floor(-diff(TP_TEMP_VAR.H.TabHeight(2:3))/2-.5) -2 0],'back',TP_TEMP_VAR.H.Color{1},'fore',TP_TEMP_VAR.H.Color{2},'enable','inactive');
    set(TP_TEMP_VAR.h,{'units'},TP_TEMP_VAR.units);
    set(TP_TEMP_VAR.H.Panel(TP_TEMP_VAR.H.Tab~=gcbo),'Visible','off');
    set(TP_TEMP_VAR.H.Panel(TP_TEMP_VAR.H.Tab==gcbo),'Visible','on');
    clear TP_TEMP_VAR;
    drawnow;
end


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over echant.
function echant_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to echant (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
gcbo=hObject;
TP_TEMP_VAR.H = get(get(gcbo,'Parent'),'UserData');
TP_TEMP_VAR.old = get(TP_TEMP_VAR.H.TopPanel,'UserData');
if TP_TEMP_VAR.H.Tab(TP_TEMP_VAR.old)~=gcbo;   
    TP_TEMP_VAR.h = [TP_TEMP_VAR.H.Main TP_TEMP_VAR.H.Panel TP_TEMP_VAR.H.TopPanel TP_TEMP_VAR.H.Tab TP_TEMP_VAR.H.TopTab];
    TP_TEMP_VAR.units = get(TP_TEMP_VAR.h,'units');
    set(TP_TEMP_VAR.h,'units','pixel');
    TP_TEMP_VAR.p = get(gcbo,'pos');
    TP_TEMP_VAR.pTP = get(TP_TEMP_VAR.H.TopPanel,'pos');
    set(TP_TEMP_VAR.H.Tab(TP_TEMP_VAR.old),'pos',[TP_TEMP_VAR.pTP(1) TP_TEMP_VAR.p(2) TP_TEMP_VAR.pTP(3) TP_TEMP_VAR.p(4)],'back',TP_TEMP_VAR.H.Color{3},'fore',TP_TEMP_VAR.H.Color{4},'enable',TP_TEMP_VAR.H.Enable);
    TP_TEMP_VAR.pP = get(TP_TEMP_VAR.H.Panel(1),'pos');
    set(TP_TEMP_VAR.H.TopPanel,'pos',[TP_TEMP_VAR.p(1) TP_TEMP_VAR.pP(4) TP_TEMP_VAR.p(3) TP_TEMP_VAR.pTP(4)],'UserData',find(TP_TEMP_VAR.H.Tab==gcbo));
    set(TP_TEMP_VAR.H.TopTab  ,'pos',[TP_TEMP_VAR.p(1)+1 TP_TEMP_VAR.pP(4) TP_TEMP_VAR.p(3)-2 1]);
    set(gcbo,'pos',TP_TEMP_VAR.p+[1 floor(-diff(TP_TEMP_VAR.H.TabHeight(2:3))/2-.5) -2 0],'back',TP_TEMP_VAR.H.Color{1},'fore',TP_TEMP_VAR.H.Color{2},'enable','inactive');
    set(TP_TEMP_VAR.h,{'units'},TP_TEMP_VAR.units);
    set(TP_TEMP_VAR.H.Panel(TP_TEMP_VAR.H.Tab~=gcbo),'Visible','off');
    set(TP_TEMP_VAR.H.Panel(TP_TEMP_VAR.H.Tab==gcbo),'Visible','on');
    clear TP_TEMP_VAR;
    drawnow;
end


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over buttontfd.
function buttontfd_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to buttontfd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
gcbo=hObject;
TP_TEMP_VAR.H = get(get(gcbo,'Parent'),'UserData');
TP_TEMP_VAR.old = get(TP_TEMP_VAR.H.TopPanel,'UserData');
if TP_TEMP_VAR.H.Tab(TP_TEMP_VAR.old)~=gcbo;   
    TP_TEMP_VAR.h = [TP_TEMP_VAR.H.Main TP_TEMP_VAR.H.Panel TP_TEMP_VAR.H.TopPanel TP_TEMP_VAR.H.Tab TP_TEMP_VAR.H.TopTab];
    TP_TEMP_VAR.units = get(TP_TEMP_VAR.h,'units');
    set(TP_TEMP_VAR.h,'units','pixel');
    TP_TEMP_VAR.p = get(gcbo,'pos');
    TP_TEMP_VAR.pTP = get(TP_TEMP_VAR.H.TopPanel,'pos');
    set(TP_TEMP_VAR.H.Tab(TP_TEMP_VAR.old),'pos',[TP_TEMP_VAR.pTP(1) TP_TEMP_VAR.p(2) TP_TEMP_VAR.pTP(3) TP_TEMP_VAR.p(4)],'back',TP_TEMP_VAR.H.Color{3},'fore',TP_TEMP_VAR.H.Color{4},'enable',TP_TEMP_VAR.H.Enable);
    TP_TEMP_VAR.pP = get(TP_TEMP_VAR.H.Panel(1),'pos');
    set(TP_TEMP_VAR.H.TopPanel,'pos',[TP_TEMP_VAR.p(1) TP_TEMP_VAR.pP(4) TP_TEMP_VAR.p(3) TP_TEMP_VAR.pTP(4)],'UserData',find(TP_TEMP_VAR.H.Tab==gcbo));
    set(TP_TEMP_VAR.H.TopTab  ,'pos',[TP_TEMP_VAR.p(1)+1 TP_TEMP_VAR.pP(4) TP_TEMP_VAR.p(3)-2 1]);
    set(gcbo,'pos',TP_TEMP_VAR.p+[1 floor(-diff(TP_TEMP_VAR.H.TabHeight(2:3))/2-.5) -2 0],'back',TP_TEMP_VAR.H.Color{1},'fore',TP_TEMP_VAR.H.Color{2},'enable','inactive');
    set(TP_TEMP_VAR.h,{'units'},TP_TEMP_VAR.units);
    set(TP_TEMP_VAR.H.Panel(TP_TEMP_VAR.H.Tab~=gcbo),'Visible','off');
    set(TP_TEMP_VAR.H.Panel(TP_TEMP_VAR.H.Tab==gcbo),'Visible','on');
    clear TP_TEMP_VAR;
    drawnow;
end


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over bfiltsig.
function bfiltsig_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to bfiltsig (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
gcbo=hObject;
TP_TEMP_VAR.H = get(get(gcbo,'Parent'),'UserData');
TP_TEMP_VAR.old = get(TP_TEMP_VAR.H.TopPanel,'UserData');
if TP_TEMP_VAR.H.Tab(TP_TEMP_VAR.old)~=gcbo;   
    TP_TEMP_VAR.h = [TP_TEMP_VAR.H.Main TP_TEMP_VAR.H.Panel TP_TEMP_VAR.H.TopPanel TP_TEMP_VAR.H.Tab TP_TEMP_VAR.H.TopTab];
    TP_TEMP_VAR.units = get(TP_TEMP_VAR.h,'units');
    set(TP_TEMP_VAR.h,'units','pixel');
    TP_TEMP_VAR.p = get(gcbo,'pos');
    TP_TEMP_VAR.pTP = get(TP_TEMP_VAR.H.TopPanel,'pos');
    set(TP_TEMP_VAR.H.Tab(TP_TEMP_VAR.old),'pos',[TP_TEMP_VAR.pTP(1) TP_TEMP_VAR.p(2) TP_TEMP_VAR.pTP(3) TP_TEMP_VAR.p(4)],'back',TP_TEMP_VAR.H.Color{3},'fore',TP_TEMP_VAR.H.Color{4},'enable',TP_TEMP_VAR.H.Enable);
    TP_TEMP_VAR.pP = get(TP_TEMP_VAR.H.Panel(1),'pos');
    set(TP_TEMP_VAR.H.TopPanel,'pos',[TP_TEMP_VAR.p(1) TP_TEMP_VAR.pP(4) TP_TEMP_VAR.p(3) TP_TEMP_VAR.pTP(4)],'UserData',find(TP_TEMP_VAR.H.Tab==gcbo));
    set(TP_TEMP_VAR.H.TopTab  ,'pos',[TP_TEMP_VAR.p(1)+1 TP_TEMP_VAR.pP(4) TP_TEMP_VAR.p(3)-2 1]);
    set(gcbo,'pos',TP_TEMP_VAR.p+[1 floor(-diff(TP_TEMP_VAR.H.TabHeight(2:3))/2-.5) -2 0],'back',TP_TEMP_VAR.H.Color{1},'fore',TP_TEMP_VAR.H.Color{2},'enable','inactive');
    set(TP_TEMP_VAR.h,{'units'},TP_TEMP_VAR.units);
    set(TP_TEMP_VAR.H.Panel(TP_TEMP_VAR.H.Tab~=gcbo),'Visible','off');
    set(TP_TEMP_VAR.H.Panel(TP_TEMP_VAR.H.Tab==gcbo),'Visible','on');
    clear TP_TEMP_VAR;
    drawnow;
end


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over bfir.
function bfir_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to bfir (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
gcbo=hObject;
TP_TEMP_VAR.H = get(get(gcbo,'Parent'),'UserData');
TP_TEMP_VAR.old = get(TP_TEMP_VAR.H.TopPanel,'UserData');
if TP_TEMP_VAR.H.Tab(TP_TEMP_VAR.old)~=gcbo;   
    TP_TEMP_VAR.h = [TP_TEMP_VAR.H.Main TP_TEMP_VAR.H.Panel TP_TEMP_VAR.H.TopPanel TP_TEMP_VAR.H.Tab TP_TEMP_VAR.H.TopTab];
    TP_TEMP_VAR.units = get(TP_TEMP_VAR.h,'units');
    set(TP_TEMP_VAR.h,'units','pixel');
    TP_TEMP_VAR.p = get(gcbo,'pos');
    TP_TEMP_VAR.pTP = get(TP_TEMP_VAR.H.TopPanel,'pos');
    set(TP_TEMP_VAR.H.Tab(TP_TEMP_VAR.old),'pos',[TP_TEMP_VAR.pTP(1) TP_TEMP_VAR.p(2) TP_TEMP_VAR.pTP(3) TP_TEMP_VAR.p(4)],'back',TP_TEMP_VAR.H.Color{3},'fore',TP_TEMP_VAR.H.Color{4},'enable',TP_TEMP_VAR.H.Enable);
    TP_TEMP_VAR.pP = get(TP_TEMP_VAR.H.Panel(1),'pos');
    set(TP_TEMP_VAR.H.TopPanel,'pos',[TP_TEMP_VAR.p(1) TP_TEMP_VAR.pP(4) TP_TEMP_VAR.p(3) TP_TEMP_VAR.pTP(4)],'UserData',find(TP_TEMP_VAR.H.Tab==gcbo));
    set(TP_TEMP_VAR.H.TopTab  ,'pos',[TP_TEMP_VAR.p(1)+1 TP_TEMP_VAR.pP(4) TP_TEMP_VAR.p(3)-2 1]);
    set(gcbo,'pos',TP_TEMP_VAR.p+[1 floor(-diff(TP_TEMP_VAR.H.TabHeight(2:3))/2-.5) -2 0],'back',TP_TEMP_VAR.H.Color{1},'fore',TP_TEMP_VAR.H.Color{2},'enable','inactive');
    set(TP_TEMP_VAR.h,{'units'},TP_TEMP_VAR.units);
    set(TP_TEMP_VAR.H.Panel(TP_TEMP_VAR.H.Tab~=gcbo),'Visible','off');
    set(TP_TEMP_VAR.H.Panel(TP_TEMP_VAR.H.Tab==gcbo),'Visible','on');
    clear TP_TEMP_VAR;
    drawnow;
end


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over biir.
function biir_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to biir (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
gcbo=hObject;
TP_TEMP_VAR.H = get(get(gcbo,'Parent'),'UserData');
TP_TEMP_VAR.old = get(TP_TEMP_VAR.H.TopPanel,'UserData');
if TP_TEMP_VAR.H.Tab(TP_TEMP_VAR.old)~=gcbo;   
    TP_TEMP_VAR.h = [TP_TEMP_VAR.H.Main TP_TEMP_VAR.H.Panel TP_TEMP_VAR.H.TopPanel TP_TEMP_VAR.H.Tab TP_TEMP_VAR.H.TopTab];
    TP_TEMP_VAR.units = get(TP_TEMP_VAR.h,'units');
    set(TP_TEMP_VAR.h,'units','pixel');
    TP_TEMP_VAR.p = get(gcbo,'pos');
    TP_TEMP_VAR.pTP = get(TP_TEMP_VAR.H.TopPanel,'pos');
    set(TP_TEMP_VAR.H.Tab(TP_TEMP_VAR.old),'pos',[TP_TEMP_VAR.pTP(1) TP_TEMP_VAR.p(2) TP_TEMP_VAR.pTP(3) TP_TEMP_VAR.p(4)],'back',TP_TEMP_VAR.H.Color{3},'fore',TP_TEMP_VAR.H.Color{4},'enable',TP_TEMP_VAR.H.Enable);
    TP_TEMP_VAR.pP = get(TP_TEMP_VAR.H.Panel(1),'pos');
    set(TP_TEMP_VAR.H.TopPanel,'pos',[TP_TEMP_VAR.p(1) TP_TEMP_VAR.pP(4) TP_TEMP_VAR.p(3) TP_TEMP_VAR.pTP(4)],'UserData',find(TP_TEMP_VAR.H.Tab==gcbo));
    set(TP_TEMP_VAR.H.TopTab  ,'pos',[TP_TEMP_VAR.p(1)+1 TP_TEMP_VAR.pP(4) TP_TEMP_VAR.p(3)-2 1]);
    set(gcbo,'pos',TP_TEMP_VAR.p+[1 floor(-diff(TP_TEMP_VAR.H.TabHeight(2:3))/2-.5) -2 0],'back',TP_TEMP_VAR.H.Color{1},'fore',TP_TEMP_VAR.H.Color{2},'enable','inactive');
    set(TP_TEMP_VAR.h,{'units'},TP_TEMP_VAR.units);
    set(TP_TEMP_VAR.H.Panel(TP_TEMP_VAR.H.Tab~=gcbo),'Visible','off');
    set(TP_TEMP_VAR.H.Panel(TP_TEMP_VAR.H.Tab==gcbo),'Visible','on');
    clear TP_TEMP_VAR;
    drawnow;
end



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function tbshan_slider2_Callback(hObject, eventdata, handles)
% hObject    handle to tbshan_slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
a=0.95;

signal_cx=[0:0.01:100];
signal_c=a.^(signal_cx);

if strcmp(get(hObject,'Type'),'uicontrol')
    slider_valuecos = get(hObject,'Value');
    set(handles.tbshan_slidervaluecos,'String',slider_valuecos);
else
    slider_valuecos=1;
end

slider_valuesin=get(handles.tbshan_slider3, 'Value');

signal_c= cos(2*pi*slider_valuecos*signal_cx) + sin(2*pi*slider_valuesin*signal_cx);
signal_cx=signal_cx(1:length(signal_c));

axes(handles.tbshan_out_t);
cla
h=plot(signal_cx,signal_c);
axis([0 50 -2 2])

slider_value=get(handles.tbshan_slider, 'Value');

Te=1/slider_value;
xte=signal_cx(1):Te:signal_cx(end);
signal_c_xte= cos(2*pi*slider_valuecos*xte) +  sin(2*pi*slider_valuesin*xte);
yte = interp1(signal_cx,signal_c,xte);
hold on, stem(xte,yte,'r');
% hold on, stem(xte,signal_c_xte,'g');

%graph frequentiels
Fe = 1/Te;
axes(handles.tbshan_out_f);
axis([-Fe/2 Fe/2 0 1.2])
axis square
cla



signal_fft=fft(signal_c_xte);
signal_fft=fft(signal_c_xte)/ numel( signal_c_xte );
FFT_T =  Fe*linspace( 0, 1, numel( signal_c_xte ))-Fe/2;

%axes(handles.tbshan_out_f);
% plot(x,y);
plot(FFT_T,fftshift(2*abs(signal_fft(1:numel(signal_c_xte)))));
% hold on, plot(x,y,'r--');
% hold on, plot(x-2*pi,y);
% hold on, plot(x+2*pi,y);
% hold on, plot(x-2*pi*slider_value,y,'r--');
% hold on, plot(x+2*pi*slider_value,y,'r--');

% --- Executes during object creation, after setting all properties.
function tbshan_slider2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tbshan_slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function tbshan_slidervaluecos_Callback(hObject, eventdata, handles)
% hObject    handle to tbshan_slidervaluecos (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tbshan_slidervaluecos as text
%        str2double(get(hObject,'String')) returns contents of tbshan_slidervaluecos as a double


% --- Executes during object creation, after setting all properties.
function tbshan_slidervaluecos_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tbshan_slidervaluecos (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function tbshan_slider3_Callback(hObject, eventdata, handles)
% hObject    handle to tbshan_slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
a=0.95;

signal_cx=[0:0.01:100];
signal_c=a.^(signal_cx);

if strcmp(get(hObject,'Type'),'uicontrol')
    slider_valuesin = get(hObject,'Value');
    set(handles.tbshan_slidervaluesin,'String',slider_valuesin);
else
    slider_valuesin=1;
end

slider_valuecos=get(handles.tbshan_slider2, 'Value');

signal_c= cos(2*pi*slider_valuecos*signal_cx) + sin(2*pi*slider_valuesin*signal_cx);
signal_cx=signal_cx(1:length(signal_c));

axes(handles.tbshan_out_t);
cla
h=plot(signal_cx,signal_c);
axis([0 50 -2 2])

slider_value=get(handles.tbshan_slider, 'Value');

Te=1/slider_value;
xte=signal_cx(1):Te:signal_cx(end);
signal_c_xte= cos(2*pi*slider_valuecos*xte) +  sin(2*pi*slider_valuesin*xte);
yte = interp1(signal_cx,signal_c,xte);
hold on, stem(xte,yte,'r');
% hold on, stem(xte,signal_c_xte,'g');

%graph frequentiels
Fe = 1/Te;
axes(handles.tbshan_out_f);
axis([-Fe/2 Fe/2 0 1.2])
axis square
cla



signal_fft=fft(signal_c_xte);
signal_fft=fft(signal_c_xte)/ numel( signal_c_xte );
FFT_T =  Fe*linspace( 0, 1, numel( signal_c_xte ))-Fe/2;

%axes(handles.tbshan_out_f);
% plot(x,y);
plot(FFT_T,fftshift(2*abs(signal_fft(1:numel(signal_c_xte)))));
% hold on, plot(x,y,'r--');
% hold on, plot(x-2*pi,y);
% hold on, plot(x+2*pi,y);
% hold on, plot(x-2*pi*slider_value,y,'r--');
% hold on, plot(x+2*pi*slider_value,y,'r--');

% --- Executes during object creation, after setting all properties.
function tbshan_slider3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tbshan_slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function tbshan_slidervaluesin_Callback(hObject, eventdata, handles)
% hObject    handle to tbshan_slidervaluesin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tbshan_slidervaluesin as text
%        str2double(get(hObject,'String')) returns contents of tbshan_slidervaluesin as a double


% --- Executes during object creation, after setting all properties.
function tbshan_slidervaluesin_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tbshan_slidervaluesin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit6_Callback(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit6 as text
%        str2double(get(hObject,'String')) returns contents of edit6 as a double


% --- Executes during object creation, after setting all properties.
function edit6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu2.
function popupmenu2_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu2


signal_cx=[0:1:100];
Fe=0.5;
Te=1/Fe;
Fc=0.05;
xte=signal_cx(1):Te:signal_cx(end);
content = get(hObject,'Value');
if content==1
    signal_c= sin(2*pi*Fc*xte);
end
if content==2
    signal_c= cos(2*pi*Fc*xte);
end
if content==3
    signal_c = zeros(1,size(xte,2));
    signal_c(floor(size(xte,2)/2))=1;
end
if content==4
    signal_c = zeros(1,size(xte,2));
    signal_c(floor(size(xte,2)/3):floor(2*size(xte,2)/3))=1;
end
if content==5
    signal_c = exp(-(xte-size(signal_cx,2)/2).*(xte-size(signal_cx,2)/2)/(2*5*5));
end
if content==6
    signal_c = sinc(2*pi*Fc*(xte-size(signal_cx,2)/2));
end

axes(handles.tbtfd_t)
h=stem(xte,signal_c);

xlabel('t');
ylabel('x(t)');

set(h,'Tag','signal_entree_tbtfd')

set(handles.tbtfd_sigN,'String',length(signal_c));
set(handles.tbtfd_inN,'String',length(signal_c));

Fe = 1/Te;
axes(handles.tbtfd_f);

axis square
cla

signal_fft=fft(signal_c)/ numel( signal_c );
FFT_T =  Fe*linspace( 0, 1, numel( signal_c ))-Fe/2;
FFT_T = FFT_T/Fe;
%axes(handles.tbshan_out_f);
% plot(x,y);


if content==3
    plot(FFT_T,fftshift(abs(numel(signal_c)*signal_fft(1:numel(signal_c)))));
    ylim([0 2]);
else
    plot(FFT_T,fftshift(2*abs(signal_fft(1:numel(signal_c)))));
    ylim('auto');
end
xlim([-1/2 1/2])
xlabel('w frequence normalisée entre -1/2 et 1/2');
ylabel('|X(w)|');

set(handles.tbtfd_f,'Xtick', -1/2:0.1:1/2);

% --- Executes during object creation, after setting all properties.
function popupmenu2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit7_Callback(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit7 as text
%        str2double(get(hObject,'String')) returns contents of edit7 as a double


% --- Executes during object creation, after setting all properties.
function edit7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit8_Callback(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit8 as text
%        str2double(get(hObject,'String')) returns contents of edit8 as a double


% --- Executes during object creation, after setting all properties.
function edit8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
