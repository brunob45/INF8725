function varargout = Sinus2D_FFT(varargin)
% SINUS2D_FFT MATLAB code for Sinus2D_FFT.fig
%      SINUS2D_FFT, by itself, creates a new SINUS2D_FFT or raises the existing
%      singleton*.
%
%      H = SINUS2D_FFT returns the handle to a new SINUS2D_FFT or the handle to
%      the existing singleton*.
%
%      SINUS2D_FFT('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SINUS2D_FFT.M with the given input arguments.
%
%      SINUS2D_FFT('Property','Value',...) creates a new SINUS2D_FFT or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Sinus2D_FFT_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Sinus2D_FFT_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Sinus2D_FFT

% Last Modified by GUIDE v2.5 28-Oct-2013 11:43:58

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Sinus2D_FFT_OpeningFcn, ...
                   'gui_OutputFcn',  @Sinus2D_FFT_OutputFcn, ...
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


% --- Executes just before Sinus2D_FFT is made visible.
function Sinus2D_FFT_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Sinus2D_FFT (see VARARGIN)

% Choose default command line output for Sinus2D_FFT
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Sinus2D_FFT wait for user response (see UIRESUME)
% uiwait(handles.MainWindow);

CreateImageAndFFT( handles );


% --- Outputs from this function are returned to the command line.
function varargout = Sinus2D_FFT_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


function CreateImageAndFFT( handles )
    Amplitude = 1;
    Period = get(findobj('Tag','Slider_Period'),'Value');
    Angle = get(findobj('Tag','Slider_Angle'),'Value');
    Phase = get(findobj('Tag','Slider_Phase'),'Value');
    ImageSize = 512;
    
    Angle = Angle * pi / 180;
    
    ImageSignal = ImageSine2( Amplitude, Angle, Period, Phase, ImageSize );
    
    axes(handles.Axe_Signal);
    imshow( ImageSignal, [0 255] );
    
    PadHeight = 2^nextpow2(ImageSize);
    PadWidth = 2^nextpow2(ImageSize);
    ImageFFT = fftshift(fft2(ImageSignal,PadHeight,PadWidth));
    
    axes(handles.Axe_FFT);
    imshow(abs(255*ImageFFT./max(max(ImageFFT))));
    
    drawnow;


% --- Executes on slider movement.
function Slider_Period_Callback(hObject, eventdata, handles)
% hObject    handle to Slider_Period (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
Period = get(hObject,'Value');
Period = ceil( Period );
set(hObject,'Value', Period);
StringVal = num2str( Period );
set( findobj('Tag','Edit_Period'), 'String', StringVal );

CreateImageAndFFT( handles );


% --- Executes during object creation, after setting all properties.
function Slider_Period_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Slider_Period (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function Edit_Period_Callback(hObject, eventdata, handles)
% hObject    handle to Edit_Period (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Edit_Period as text
%        str2double(get(hObject,'String')) returns contents of Edit_Period as a double

% --- Executes during object creation, after setting all properties.
function Edit_Period_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Edit_Period (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function Slider_Phase_Callback(hObject, eventdata, handles)
% hObject    handle to Slider_Period (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
PhaseW = get(hObject,'Value');
StringVal = num2str( PhaseW );
set( findobj('Tag','Edit_Phase'), 'String', StringVal );

CreateImageAndFFT( handles );


% --- Executes during object creation, after setting all properties.
function Slider_Phase_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Slider_Period (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function Edit_Angle_Callback(hObject, eventdata, handles)
% hObject    handle to Edit_Angle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Edit_Angle as text
%        str2double(get(hObject,'String')) returns contents of Edit_Angle as a double


% --- Executes during object creation, after setting all properties.
function Edit_Angle_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Edit_Angle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function Slider_Angle_Callback(hObject, eventdata, handles)
% hObject    handle to Slider_Angle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
Angle = get(hObject,'Value');
Angle = ceil( Angle );
set(hObject,'Value', Angle);
StringVal = num2str( Angle );
set( findobj('Tag','Edit_Angle'), 'String', StringVal );

CreateImageAndFFT( handles );

% --- Executes during object creation, after setting all properties.
function Slider_Angle_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Slider_Angle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function Edit_Phase_Callback(hObject, eventdata, handles)
% hObject    handle to Edit_Phase (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Edit_Phase as text
%        str2double(get(hObject,'String')) returns contents of Edit_Phase as a double


% --- Executes during object creation, after setting all properties.
function Edit_Phase_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Edit_Phase (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
