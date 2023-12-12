
%% Gaze360 - Azimuth angle and AOI angles

clear all
close all
clc

for who = 1:1
    
%% ------ Hyperparameters ------
geometrical = 'True';
who = 2;
interpolation = true;
gazeoffsets = false;
percentagebox = 0.5;
sigma = 2;
%% Check values for multiple widths 
 
% first for child, second for therapist
% per la persona, per cambiare l'AOI del terapista devo modificare la width
% del child

width_k = [0.30, 0.20];
width_nao = [0.30, 0.20];
width_p = [0.15, 0.15];
width_dx = [0.15, 0.15];
width_sx = [0.15, 0.15];
width_dietro = [0.15, 0.15];

width_ky = [0.20, 0.10];
width_naoy = [0.05,0.05];
width_py = [0.10, 0.20];
width_dx_y = [0.10, 0.10];
width_sx_y = [0.10, 0.10];
width_dietro_y = [0.15, 0.15];

width_kz = [0.20, 0.10];
width_naoz = [0.05, 0.05];
width_pz = [0.10, 0.20];
width_dx_z = [0.05, 0.05];
width_sx_z = [0.05, 0.05];
width_dietro_z = [0.05, 0.05];

width_x = {};
width_x{1} = width_k;
width_x{2} = width_nao;
width_x{3} = width_dx;
width_x{4} = width_sx;
width_x{5} = width_dietro;
width_x{6} = width_p;

width_y = {};
width_y{1} = width_ky;
width_y{2} = width_naoy;
width_y{3} = width_dx_y;
width_y{4} = width_sx_y;
width_y{5} = width_dietro_y;
width_y{6} = width_py;

width_z = {};
width_z{1} = width_kz;
width_z{2} = width_naoz;
width_z{3} = width_dx_z;
width_z{4} = width_sx_z;
width_z{5} = width_dietro_z;
width_z{6} = width_pz;

%% Read Scene characteristics file

% visto dal soggetto: sx ha x>0 

kinectcoord = [0.0,0.0,0.0];
naocoord = [0.45,-0.12,1.35]; %sul tavolo
% naocoord = [0.35,-0.85,0.99]; % per terra
dx_coord = [-1.38,0.60,1.42];
sx_coord = [1.50,0.40,1.45];
dietro_coord = [0.06,0.08, 3.38];

target_label = {'kinect','robot','poster_dx','poster_sx','poster_elsewhere','altra_p','rabbit','therapist','Nowhere'};

coord = {};
coord{1} = kinectcoord;
coord{2} = naocoord;
coord{3} = dx_coord;
coord{4} = sx_coord;
coord{5} = dietro_coord;
numbertarget = size(coord,2)+3;

% Relative path to the patients directory
direct = '../data/patients/no_robot_therapist';

% Get a list of all files and folders in this directory
files = dir(direct);
disp("before");
disp(files);

% Filter out the entries that are directories
subFolders = files([files.isdir]);

% direct = '/Users/lostinstaches/Desktop/Gabriele_Laura';

% folders={1};
% subjects = {'p1'};
vector ={'1'};
% 
% extot = {};
% elevationtot = {};
% tmtot = {};
% 
% child_az = {};
% adult_az = {};
% timtot = {};
% 
% descripttot = {};
% exvectot = {};
% 
% sessions_all = [1];
% parents_all = {1};

% Initialize storage variables
extot = {};
elevationtot = {};
tmtot = {};
child_az = {};
adult_az = {};
timtot = {};
descripttot = {};
exvectot = {};
sessions_all = [];
parents_all = {};

% Gaze360, NAO_standard, Subject/Therapist_standard angles


% for patient = 1:size(subjects)
% Loop through each subfolder (patient)
for k = 1:length(subFolders)
    close all;
    folderName = subFolders(k).name;
    
    % Skip the '.' and '..' directories
    if strcmp(folderName, '.') || strcmp(folderName, '..')
        continue;
    end

    % Display the current processing folder (patient)
    disp(['Processing patient: ', folderName]);

    % Initialize session data for the current patient
    azimutessession = {};
    elevationsession = {};
    tempossession = {};
    descriptsession = {};
    keypsession = {};
    exvecsession = {};

    % Process each session for the current patient
    % Assuming only one session per patient for simplicity


    [keyp, tempos, azimutes, elevations, ...
        confidence_az_th, confidence_az_ch, confidence_el_th, confidence_el_ch] = ...
        Read_files_elevation_Odi_solo_porta_2pp(direct, folderName, interpolation, ...
        vector{1}, percentagebox, coord); % Assuming vector{1} is applicable to all patients

    % Store session data
    azimutessession{end+1} = azimutes;
    elevationsession{end+1} = elevations;
    tempossession{end+1} = tempos;
    keypsession{end+1} = keyp;

    % Check for empty data and continue to the next patient if needed
    if isempty(azimutessession) || isempty(azimutessession{1})
        continue;
    end

    % Store patient data
    extot{end+1} = azimutessession;
    elevationtot{end+1} = elevationsession;
    tmtot{end+1} = tempossession;
    descripttot{end+1} = descriptsession;
    exvectot{end+1} = exvecsession;
end

end