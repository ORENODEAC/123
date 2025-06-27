// ===========================
// COLMAP 視覺化器 main.js for "指定組上傳 AlignData.txt" (完整版)
// ===========================

// ===== 分頁切換 =====
function switchTab(tab) {
    document.getElementById('tab-calc-btn').classList.toggle('active', tab === 'calc');
    document.getElementById('tab-view-btn').classList.toggle('active', tab === 'view');
    document.getElementById('tab-calc-content').classList.toggle('hidden', tab !== 'calc');
    document.getElementById('tab-view-content').classList.toggle('hidden', tab !== 'view');
    if(tab === 'view') setTimeout(resizeCanvas, 0);
}
switchTab('calc');

// ===== 全域變數與UI =====
let groups = []; // [{ images, points, alignData, useAlign, label, visible }]
let pointsBuffers = [], colorsBuffers = [], cameraFrustumBuffers = [], cameraFrustumColorBuffers = [], cameraFrustumLineCounts = [];
let colorGroupMode = true;

const camColors = [
    [1, 1, 0],[1, 0.4, 0.4],[0, 1, 1],[0.3, 1, 0.3],[0.6, 0.5, 1],[1, 0.5, 1],[1, 0.8, 0.5],[0.7, 1, 0.7]
];

const colorModeToggle = document.getElementById('colorModeToggle');
const cloudVisibilityList = document.getElementById('cloudVisibilityList');
const dropZone = document.getElementById('dropZone');
const loading = document.getElementById('loading');
const imagesStatus = document.getElementById('imagesStatus');
const pointsStatus = document.getElementById('pointsStatus');
const alignStatus = document.getElementById('alignStatus');
const stats = document.getElementById('stats');
colorModeToggle.checked = true;
colorModeToggle.addEventListener('change', function() {
    colorGroupMode = colorModeToggle.checked;
    setupBuffers(); render();
});

// ====== 檔案拖曳、上傳（多組自動分組，AlignData.txt需手動上傳至組） ======
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('dragover'); });
dropZone.addEventListener('drop', (e) => {
    e.preventDefault(); dropZone.classList.remove('dragover');
    handleFiles(Array.from(e.dataTransfer.files));
});

async function handleFiles(files) {
    loading.style.display = 'block';
    // 只配對 images/points3D，不處理 AlignData.txt
    let fileMap = {};
    for (const file of files) {
        let base = file.name.replace(/(\.txt)$/i, '').toLowerCase();
        let groupKey = base.replace(/(images|points3d)$/i, '').replace(/[\W_]+$/,'') || "g" + groups.length;
        if (!fileMap[groupKey]) fileMap[groupKey] = {};
        if (/images\.txt$/i.test(file.name)) fileMap[groupKey].images = file;
        if (/points3d\.txt$/i.test(file.name)) fileMap[groupKey].points = file;
    }
    for (const groupKey in fileMap) {
        const filesObj = fileMap[groupKey];
        let label = groupKey;
        let group = { images: null, points: null, alignData: null, useAlign: false, label, visible: true };
        if (filesObj.images) group.images = parseImagesFile(await filesObj.images.text());
        if (filesObj.points) group.points = parsePointsFile(await filesObj.points.text());
        groups.push(group);
    }
    setupBuffers();
    updateCloudVisibilityList();
    focusOnScene();
    render();
    updateStats();
    loading.style.display = 'none';
}

// --------- 解析/轉換 ---------
function parseImagesFile(text) {
    const lines = text.split('\n').filter(line => line.trim() && !line.startsWith('#'));
    const images = [];
    for (let i = 0; i < lines.length; i += 2) {
        const parts = lines[i].trim().split(/\s+/);
        if (parts.length >= 10) {
            const qw = parseFloat(parts[1]), qx = parseFloat(parts[2]), qy = parseFloat(parts[3]), qz = parseFloat(parts[4]);
            const tx = parseFloat(parts[5]), ty = parseFloat(parts[6]), tz = parseFloat(parts[7]);
            images.push({
                id: parseInt(parts[0]),
                quaternion: [qw, qx, qy, qz],
                translation: [tx, ty, tz],
                name: parts[9]
            });
        }
    }
    return images;
}
function parsePointsFile(text) {
    const lines = text.split('\n').filter(line => line.trim() && !line.startsWith('#'));
    const points = [];
    for (const line of lines) {
        const parts = line.trim().split(/\s+/);
        if (parts.length >= 7) {
            const x = parseFloat(parts[1]), y = parseFloat(parts[2]), z = parseFloat(parts[3]);
            const r = parseInt(parts[4]) / 255.0, g = parseInt(parts[5]) / 255.0, b = parseInt(parts[6]) / 255.0;
            points.push({ position:[x,y,z], color:[r,g,b] });
        }
    }
    return points;
}
function parseAlignData(txt) {
    let lines = txt.split('\n').map(s => s.trim()).filter(s => s.length > 0 && !s.startsWith('#'));
    let scale = 1.0;
    let R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    let t = [0, 0, 0];
    for (let i = 0; i < lines.length; i++) {
        if (lines[i].startsWith("scale")) {
            let parts = lines[i].split(/\s+/);
            scale = parseFloat(parts[1] || lines[i + 1]); // 支援: "scale x.xx" 或下一行
            if (isNaN(scale) && i + 1 < lines.length) {
                scale = parseFloat(lines[++i]);
            }
        }
        if (lines[i] === "R" || lines[i].toLowerCase().includes('rotation')) {
            R = [
                lines[i + 1].split(/\s+/).map(Number),
                lines[i + 2].split(/\s+/).map(Number),
                lines[i + 3].split(/\s+/).map(Number)
            ];
            i += 3;
        }
        if (lines[i] === "t" || lines[i].toLowerCase().includes('translation')) {
            t = lines[i + 1].split(/\s+/).map(Number);
            i += 1;
        }
        // 支援最簡格式: [scale, R1, R2, R3, t]
        if (!isNaN(parseFloat(lines[i])) && i + 4 < lines.length) {
            let s = parseFloat(lines[i]);
            let r1 = lines[i + 1].split(/\s+/).map(Number);
            let r2 = lines[i + 2].split(/\s+/).map(Number);
            let r3 = lines[i + 3].split(/\s+/).map(Number);
            let tt = lines[i + 4].split(/\s+/).map(Number);
            if (
                r1.length === 3 && r2.length === 3 && r3.length === 3 &&
                tt.length === 3 && !isNaN(s)
            ) {
                scale = s;
                R = [r1, r2, r3];
                t = tt;
                break;
            }
        }
    }
    return { scale, R, t };
}
function applyAlignToPoint(pt, group) {
    if (!group.alignData || !group.useAlign) return pt;
    let {scale, R, t} = group.alignData;
    let v = pt.position;
    let v2 = [
        scale*(R[0][0]*v[0]+R[0][1]*v[1]+R[0][2]*v[2]) + t[0],
        scale*(R[1][0]*v[0]+R[1][1]*v[1]+R[1][2]*v[2]) + t[1],
        scale*(R[2][0]*v[0]+R[2][1]*v[1]+R[2][2]*v[2]) + t[2]
    ];
    return Object.assign({}, pt, {position: v2});
}
function applyAlignToCamera(cam, group) {
    if (!group.alignData || !group.useAlign) return cam;
    let {scale, R, t} = group.alignData;
    let q = cam.quaternion, tr = cam.translation;
    let R0 = quatToMatrix3x3(q);
    let Rt = transpose3x3(R0);
    let C = [
        -(Rt[0][0]*tr[0]+Rt[0][1]*tr[1]+Rt[0][2]*tr[2]),
        -(Rt[1][0]*tr[0]+Rt[1][1]*tr[1]+Rt[1][2]*tr[2]),
        -(Rt[2][0]*tr[0]+Rt[2][1]*tr[1]+Rt[2][2]*tr[2])
    ];
    let C2 = [
        scale*(R[0][0]*C[0]+R[0][1]*C[1]+R[0][2]*C[2]) + t[0],
        scale*(R[1][0]*C[0]+R[1][1]*C[1]+R[1][2]*C[2]) + t[1],
        scale*(R[2][0]*C[0]+R[2][1]*C[1]+R[2][2]*C[2]) + t[2]
    ];
    let R_new = matmul33(R, R0);
    let t_new = [
        -(R_new[0][0]*C2[0] + R_new[0][1]*C2[1] + R_new[0][2]*C2[2]),
        -(R_new[1][0]*C2[0] + R_new[1][1]*C2[1] + R_new[1][2]*C2[2]),
        -(R_new[2][0]*C2[0] + R_new[2][1]*C2[1] + R_new[2][2]*C2[2])
    ];
    let q_new = rotmat2quat(R_new);
    return Object.assign({}, cam, {quaternion: q_new, translation: t_new});
}
function quatToMatrix3x3(q) {
    const [w, x, y, z] = q;
    return [
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ];
}
function transpose3x3(M) { return [[M[0][0],M[1][0],M[2][0]],[M[0][1],M[1][1],M[2][1]],[M[0][2],M[1][2],M[2][2]]]; }
function matmul33(A,B){
    let r = [];
    for(let i=0;i<3;i++){
        r[i]=[];
        for(let j=0;j<3;j++)
            r[i][j]=A[i][0]*B[0][j]+A[i][1]*B[1][j]+A[i][2]*B[2][j];
    }
    return r;
}
function rotmat2quat(R) {
    let tr = R[0][0] + R[1][1] + R[2][2], q = [0,0,0,0];
    if(tr > 0) {
        let s = Math.sqrt(tr+1.0)*2;
        q[0] = 0.25*s;
        q[1] = (R[2][1] - R[1][2])/s;
        q[2] = (R[0][2] - R[2][0])/s;
        q[3] = (R[1][0] - R[0][1])/s;
    } else {
        let i = [R[0][0], R[1][1], R[2][2]].indexOf(Math.max(R[0][0], R[1][1], R[2][2]));
        if(i === 0) {
            let s = Math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2])*2;
            q[0] = (R[2][1] - R[1][2])/s;
            q[1] = 0.25*s;
            q[2] = (R[0][1] + R[1][0])/s;
            q[3] = (R[0][2] + R[2][0])/s;
        } else if(i === 1) {
            let s = Math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2])*2;
            q[0] = (R[0][2] - R[2][0])/s;
            q[1] = (R[0][1] + R[1][0])/s;
            q[2] = 0.25*s;
            q[3] = (R[1][2] + R[2][1])/s;
        } else {
            let s = Math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1])*2;
            q[0] = (R[1][0] - R[0][1])/s;
            q[1] = (R[0][2] + R[2][0])/s;
            q[2] = (R[1][2] + R[2][1])/s;
            q[3] = 0.25*s;
        }
    }
    let norm = Math.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2);
    return q.map(x => x / norm);
}

// ====== WebGL 初始化與繪製 ======
const canvas = document.getElementById('canvas');
const gl = canvas.getContext('webgl');
if (!gl) { alert('您的瀏覽器不支援 WebGL'); }
function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    gl.viewport(0, 0, canvas.width, canvas.height);
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

const vertexShaderSource = `
    attribute vec3 a_position;
    attribute vec3 a_color;
    uniform mat4 u_modelViewMatrix;
    uniform mat4 u_projectionMatrix;
    uniform float u_pointSize;
    varying vec3 v_color;
    void main() {
        gl_Position = u_projectionMatrix * u_modelViewMatrix * vec4(a_position, 1.0);
        gl_PointSize = u_pointSize;
        v_color = a_color;
    }
`;
const fragmentShaderSource = `
    precision mediump float;
    varying vec3 v_color;
    void main() {
        gl_FragColor = vec4(v_color, 1.0);
    }
`;
function createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source); gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('著色器編譯錯誤:', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader); return null;
    }
    return shader;
}
function createProgram(gl, vertexShader, fragmentShader) {
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader); gl.attachShader(program, fragmentShader); gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error('程式連結錯誤:', gl.getProgramInfoLog(program));
        gl.deleteProgram(program); return null;
    }
    return program;
}
const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
const program = createProgram(gl, vertexShader, fragmentShader);
const positionLocation = gl.getAttribLocation(program, 'a_position');
const colorLocation = gl.getAttribLocation(program, 'a_color');
const modelViewMatrixLocation = gl.getUniformLocation(program, 'u_modelViewMatrix');
const projectionMatrixLocation = gl.getUniformLocation(program, 'u_projectionMatrix');
const pointSizeLocation = gl.getUniformLocation(program, 'u_pointSize');

// ====== 互動控制 & 相機 ======
function identity4() {
    return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
}
function multiplyMatrix4(a, b) {
    const out = new Array(16);
    for (let i=0; i<4; ++i) for (let j=0; j<4; ++j) {
        out[i*4+j] = 0;
        for (let k=0; k<4; ++k) out[i*4+j] += a[i*4+k]*b[k*4+j];
    }
    return out;
}
function rotateY(angle) {
    const c = Math.cos(angle), s = Math.sin(angle);
    return [c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1];
}
function rotateX(angle) {
    const c = Math.cos(angle), s = Math.sin(angle);
    return [1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1];
}
function mat4vec3mul(m, v) {
    return [
        m[0]*v[0] + m[1]*v[1] + m[2]*v[2],
        m[4]*v[0] + m[5]*v[1] + m[6]*v[2],
        m[8]*v[0] + m[9]*v[1] + m[10]*v[2]
    ];
}
let camera = {
    target: [0,0,0],
    distance: 10,
    rotation: identity4()
};
let interaction = {
    isDragging: false, isMiddleDragging: false, isShiftDragging: false,
    lastMouseX: 0, lastMouseY: 0,
    sensitivity: 0.015,
    panSensitivity: 0.005,
    zoomSensitivity: 0.1
};
canvas.addEventListener('mousedown', (e) => {
    if (e.button === 0) {
        if (e.shiftKey) interaction.isShiftDragging = true;
        else interaction.isDragging = true;
    } else if (e.button === 1) interaction.isMiddleDragging = true;
    else if (e.button === 2 && e.shiftKey) interaction.isMiddleDragging = true;
    interaction.lastMouseX = e.clientX;
    interaction.lastMouseY = e.clientY;
    e.preventDefault();
});
canvas.addEventListener('mousemove', (e) => {
    const dx = e.clientX - interaction.lastMouseX;
    const dy = e.clientY - interaction.lastMouseY;
    if (interaction.isDragging) {
        camera.rotation = multiplyMatrix4(rotateY(-dx * interaction.sensitivity), camera.rotation);
        camera.rotation = multiplyMatrix4(camera.rotation, rotateX(-dy * interaction.sensitivity));
        render();
    } else if (interaction.isMiddleDragging || interaction.isShiftDragging) {
        let right = mat4vec3mul(camera.rotation, [1,0,0]);
        let up = mat4vec3mul(camera.rotation, [0,1,0]);
        const panSpeed = camera.distance * interaction.panSensitivity;
        camera.target[0] -= right[0] * dx * panSpeed;
        camera.target[1] -= right[1] * dx * panSpeed;
        camera.target[2] -= right[2] * dx * panSpeed;
        camera.target[0] += up[0] * dy * panSpeed;
        camera.target[1] += up[1] * dy * panSpeed;
        camera.target[2] += up[2] * dy * panSpeed;
        render();
    }
    interaction.lastMouseX = e.clientX;
    interaction.lastMouseY = e.clientY;
});
canvas.addEventListener('mouseup', (e) => {
    interaction.isDragging = false;
    interaction.isMiddleDragging = false;
    interaction.isShiftDragging = false;
});
canvas.addEventListener('wheel', (e) => {
    const zoomFactor = 1 + (e.deltaY > 0 ? interaction.zoomSensitivity : -interaction.zoomSensitivity);
    camera.distance = Math.max(0.1, Math.min(100, camera.distance * zoomFactor));
    render();
    e.preventDefault();
});
canvas.addEventListener('contextmenu', (e) => e.preventDefault());
document.addEventListener('keydown', (e) => {
    switch(e.key.toLowerCase()) {
        case 'r': camera.rotation=identity4();camera.distance=10;camera.target=[0,0,0];render();break;
        case 'f': if(groups.length && totalPoints()>0) focusOnScene();break;
        case '1': camera.rotation=identity4();render();break;
        case '3': camera.rotation=rotateY(Math.PI/2);render();break;
        case '7': camera.rotation=rotateX(-Math.PI/2);render();break;
    }
});
function getCameraMatrix() {
    let forward = mat4vec3mul(camera.rotation, [0,0,1]);
    let eye = [
        camera.target[0] - forward[0]*camera.distance,
        camera.target[1] - forward[1]*camera.distance,
        camera.target[2] - forward[2]*camera.distance
    ];
    let up = mat4vec3mul(camera.rotation, [0,1,0]);
    return createLookAtMatrix(eye, camera.target, up);
}
function createLookAtMatrix(eye, center, up) {
    let z = normalize(subtract(eye, center));
    let x = normalize(cross(up, z));
    let y = cross(z, x);
    return new Float32Array([
        x[0], y[0], z[0], 0,
        x[1], y[1], z[1], 0,
        x[2], y[2], z[2], 0,
        -dot(x, eye), -dot(y, eye), -dot(z, eye), 1
    ]);
}
function normalize(v) { const l=Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); return l>0?[v[0]/l,v[1]/l,v[2]/l]:[0,0,0]; }
function subtract(a, b) { return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]; }
function cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function dot(a, b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }

function totalPoints() {
    let n = 0;
    for (let i=0; i<groups.length; ++i)
        if (groups[i].visible && groups[i].points) n += groups[i].points.length;
    return n;
}

// ====== 緩衝區與繪製 ======
function quatToMatrix(q) {
    const [w, x, y, z] = q;
    return [
        1 - 2*y*y - 2*z*z,   2*x*y - 2*z*w,       2*x*z + 2*y*w,
        2*x*y + 2*z*w,       1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w,
        2*x*z - 2*y*w,       2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y
    ];
}
function transpose3(m) { return [m[0],m[3],m[6],m[1],m[4],m[7],m[2],m[5],m[8]]; }
function mat3mulv(m, v) {
    return [
        m[0]*v[0]+m[1]*v[1]+m[2]*v[2],
        m[3]*v[0]+m[4]*v[1]+m[5]*v[2],
        m[6]*v[0]+m[7]*v[1]+m[8]*v[2]
    ];
}
function addv(a, b) { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
function getCameraFrustumVertices(q, t, scale=0.3, fov=0.9) {
    const R = quatToMatrix(q);
    const Rt = transpose3(R);
    const C = [
        -(Rt[0]*t[0] + Rt[1]*t[1] + Rt[2]*t[2]),
        -(Rt[3]*t[0] + Rt[4]*t[1] + Rt[5]*t[2]),
        -(Rt[6]*t[0] + Rt[7]*t[1] + Rt[8]*t[2])
    ];
    const depth = scale, aspect = 4/3;
    const halfH = Math.tan(fov/2) * depth, halfW = halfH * aspect;
    const corners = [
        [ halfW,  halfH, depth],
        [-halfW,  halfH, depth],
        [-halfW, -halfH, depth],
        [ halfW, -halfH, depth]
    ];
    for(let i=0; i<corners.length; ++i)
        corners[i] = addv(mat3mulv(Rt, corners[i]), C);
    return { C, corners };
}
function setupBuffers() {
    pointsBuffers = [];
    colorsBuffers = [];
    cameraFrustumBuffers = [];
    cameraFrustumColorBuffers = [];
    cameraFrustumLineCounts = [];
    for (let i = 0; i < groups.length; ++i) {
        const group = groups[i];
        // 點雲
        if (group.points && group.points.length) {
            const positions = [], colors = [];
            let groupColor = camColors[i % camColors.length];
            for (const point of group.points) {
                let p2 = applyAlignToPoint(point, group);
                positions.push(...p2.position);
                if (colorGroupMode) {
                    colors.push(...groupColor);
                } else {
                    colors.push(...point.color);
                }
            }
            const pbuf = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, pbuf);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
            pointsBuffers.push(pbuf);

            const cbuf = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, cbuf);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
            colorsBuffers.push(cbuf);
        } else {
            pointsBuffers.push(null);
            colorsBuffers.push(null);
        }
        // 相機
        if (group.images && group.images.length) {
            let camLines = [], camLineColors = [];
            const color = camColors[i % camColors.length];
            const edgeColor = color;
            const faceColor = color.map(c => c * 0.7);
            for (const img of group.images) {
                const cam2 = applyAlignToCamera(img, group);
                const { C, corners } = getCameraFrustumVertices(cam2.quaternion, cam2.translation);
                for (let j=0; j<4; ++j) {
                    camLines.push(...C); camLines.push(...corners[j]);
                    camLineColors.push(...edgeColor); camLineColors.push(...edgeColor);
                }
                for (let j=0; j<4; ++j) {
                    camLines.push(...corners[j]); camLines.push(...corners[(j+1)%4]);
                    camLineColors.push(...faceColor); camLineColors.push(...faceColor);
                }
            }
            const fbuf = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, fbuf);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(camLines), gl.STATIC_DRAW);
            cameraFrustumBuffers.push(fbuf);

            const fcolor = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, fcolor);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(camLineColors), gl.STATIC_DRAW);
            cameraFrustumColorBuffers.push(fcolor);

            cameraFrustumLineCounts.push(camLines.length / 3);
        } else {
            cameraFrustumBuffers.push(null);
            cameraFrustumColorBuffers.push(null);
            cameraFrustumLineCounts.push(0);
        }
    }
}
function updateCloudVisibilityList() {
    cloudVisibilityList.innerHTML = '';
    for (let i = 0; i < groups.length; ++i) {
        const group = groups[i];
        const item = document.createElement('li');
        item.className = 'cloud-visibility-item';

        // 色環
        const colorBall = document.createElement('span');
        colorBall.className = 'cloud-label-color';
        const c = camColors[i % camColors.length];
        colorBall.style.background = `rgb(${Math.round(c[0]*255)},${Math.round(c[1]*255)},${Math.round(c[2]*255)})`;

        // 標籤
        const label = document.createElement('span');
        label.className = 'cloud-label';
        label.textContent = group.label;

        // 顯示開關
        const switchLabel = document.createElement('label');
        switchLabel.className = 'cloud-visibility-switch';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = group.visible;
        checkbox.addEventListener('change', () => {
            group.visible = checkbox.checked;
            render();
            updateStats();
        });
        const slider = document.createElement('span');
        slider.className = 'cloud-visibility-slider';
        switchLabel.appendChild(checkbox);
        switchLabel.appendChild(slider);

        // align 勾選
        const alignSwitchLabel = document.createElement('label');
        alignSwitchLabel.className = 'cloud-visibility-switch';
        const alignCheckbox = document.createElement('input');
        alignCheckbox.type = 'checkbox';
        alignCheckbox.checked = group.useAlign;
        alignCheckbox.disabled = !group.alignData;
        alignCheckbox.title = group.alignData ? "套用 AlignData" : "請先上傳 AlignData.txt";
        alignCheckbox.addEventListener('change', () => {
            group.useAlign = alignCheckbox.checked;
            setupBuffers(); render();
        });
        const alignSlider = document.createElement('span');
        alignSlider.className = 'cloud-visibility-slider';
        alignSwitchLabel.appendChild(alignCheckbox);
        alignSwitchLabel.appendChild(alignSlider);

        // -------- align 檔案上傳按鈕 --------
        const alignUploader = document.createElement('input');
        alignUploader.type = 'file';
        alignUploader.accept = '.txt';
        alignUploader.style.marginLeft = '8px';
        alignUploader.title = '為此組上傳 AlignData.txt';
        alignUploader.addEventListener('change', async (e) => {
            if (!e.target.files.length) return;
            const file = e.target.files[0];
            const text = await file.text();
            group.alignData = parseAlignData(text);
            group.useAlign = true;
            updateCloudVisibilityList();
            setupBuffers(); render();
        });

        item.appendChild(switchLabel);
        item.appendChild(colorBall);
        item.appendChild(label);
        item.appendChild(alignSwitchLabel);
        item.appendChild(alignUploader);

        cloudVisibilityList.appendChild(item);
    }
}
function focusOnScene() {
    if (!groups.length) return;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
    let hasPoint = false;
    for (let i = 0; i < groups.length; ++i) {
        if (!groups[i].visible || !groups[i].points) continue;
        for (const point of groups[i].points) {
            let p2 = applyAlignToPoint(point, groups[i]);
            const [x, y, z] = p2.position;
            minX = Math.min(minX, x); maxX = Math.max(maxX, x);
            minY = Math.min(minY, y); maxY = Math.max(maxY, y);
            minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
            hasPoint = true;
        }
    }
    if (!hasPoint) return;
    camera.target = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2];
    const sizeX = maxX - minX, sizeY = maxY - minY, sizeZ = maxZ - minZ;
    const maxSize = Math.max(sizeX, sizeY, sizeZ);
    camera.distance = maxSize * 1.5;
    render();
}
function updateStats() {
    let statsText = '';
    for (let i = 0; i < groups.length; ++i) {
        if (groups[i].points)
            statsText += `<span style="color:${groups[i].visible ? '#fff':'#aaa'}">點雲 #${i+1}（${groups[i].label}）: ${groups[i].points.length.toLocaleString()} 個點</span><br>`;
        if (groups[i].images)
            statsText += `<span style="color:${groups[i].visible ? '#fff':'#aaa'}">相機 #${i+1}（${groups[i].label}）: ${groups[i].images.length} 個位置</span><br>`;
    }
    stats.innerHTML = statsText || "尚未載入資料";
    imagesStatus.textContent = groups.filter(g=>g.images).length > 0 ? `已載入 ${groups.filter(g=>g.images).length} 組 images` : "Images: 未載入";
    imagesStatus.className = groups.filter(g=>g.images).length > 0 ? 'status-item status-loaded' : 'status-item status-pending';
    pointsStatus.textContent = groups.filter(g=>g.points).length > 0 ? `已載入 ${groups.filter(g=>g.points).length} 組 points3D` : "Points: 未載入";
    pointsStatus.className = groups.filter(g=>g.points).length > 0 ? 'status-item status-loaded' : 'status-item status-pending';
    alignStatus.textContent = groups.filter(g=>g.alignData).length > 0 ? `AlignData: 已載入${groups.filter(g=>g.alignData).length}組` : "AlignData: 未載入";
    alignStatus.className = groups.filter(g=>g.alignData).length > 0 ? 'status-item status-loaded' : 'status-item status-pending';
}
function createPerspectiveMatrix(fov, aspect, near, far) {
    const f = Math.tan(Math.PI * 0.5 - 0.5 * fov);
    const rangeInv = 1.0 / (near - far);
    return new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (near + far) * rangeInv, -1,
        0, 0, near * far * rangeInv * 2, 0
    ]);
}
function render() {
    gl.clearColor(0.1, 0.1, 0.15, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.useProgram(program);
    const aspect = canvas.width / canvas.height;
    const projectionMatrix = createPerspectiveMatrix(Math.PI / 4, aspect, 0.1, 250);
    gl.uniformMatrix4fv(projectionMatrixLocation, false, projectionMatrix);
    const modelViewMatrix = getCameraMatrix();
    gl.uniformMatrix4fv(modelViewMatrixLocation, false, modelViewMatrix);

    gl.uniform1f(pointSizeLocation, 2.0);
    for (let i = 0; i < groups.length; ++i) {
        const group = groups[i];
        if (!group.visible || !group.points || !pointsBuffers[i]) continue;
        if (!group.points.length) continue;
        gl.bindBuffer(gl.ARRAY_BUFFER, pointsBuffers[i]);
        gl.enableVertexAttribArray(positionLocation);
        gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, colorsBuffers[i]);
        gl.enableVertexAttribArray(colorLocation);
        gl.vertexAttribPointer(colorLocation, 3, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.POINTS, 0, group.points.length);
    }
    gl.uniform1f(pointSizeLocation, 1.0);
    for (let i = 0; i < groups.length; ++i) {
        const group = groups[i];
        if (!group.visible || !group.images || !cameraFrustumBuffers[i]) continue;
        if (!cameraFrustumLineCounts[i]) continue;
        gl.bindBuffer(gl.ARRAY_BUFFER, cameraFrustumBuffers[i]);
        gl.enableVertexAttribArray(positionLocation);
        gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, cameraFrustumColorBuffers[i]);
        gl.enableVertexAttribArray(colorLocation);
        gl.vertexAttribPointer(colorLocation, 3, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.LINES, 0, cameraFrustumLineCounts[i]);
    }
    updateViewCube();
}

// ====== ViewCube 控制器 ======
const viewcube = document.getElementById('viewcube');
const vctx = viewcube.getContext('2d');
let viewcubeDragging = false;
let viewcubeLastX = 0, viewcubeLastY = 0;
let viewcubeLastRot = identity4();
function drawViewCube(rot) {
    vctx.clearRect(0, 0, 80, 80);
    const size = 24, cx = 40, cy = 40;
    const verts = [
        [-1,-1,-1], [ 1,-1,-1], [ 1, 1,-1], [-1, 1,-1],
        [-1,-1, 1], [ 1,-1, 1], [ 1, 1, 1], [-1, 1, 1]
    ];
    function mat4vec3mulV(m, v) {
        return [
            m[0]*v[0] + m[1]*v[1] + m[2]*v[2],
            m[4]*v[0] + m[5]*v[1] + m[6]*v[2],
            m[8]*v[0] + m[9]*v[1] + m[10]*v[2]
        ];
    }
    const pts = verts.map(p => {
        let v = mat4vec3mulV(rot, p);
        return [cx + v[0]*size, cy - v[1]*size];
    });
    const edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ];
    vctx.strokeStyle = "#fff"; vctx.lineWidth = 2;
    for(const [a,b] of edges){
        vctx.beginPath(); vctx.moveTo(pts[a][0],pts[a][1]); vctx.lineTo(pts[b][0],pts[b][1]); vctx.stroke();
    }
    vctx.font = "bold 12px sans-serif";
    vctx.textAlign = "center";
    vctx.textBaseline = "middle";
    const faces = [
        {label:"F", idxs:[0,1,2,3], color:"#4CAF50", center:[0,0,-1]},
        {label:"B", idxs:[4,5,6,7], color:"#e91e63", center:[0,0,1]},
        {label:"L", idxs:[0,3,7,4], color:"#03a9f4", center:[-1,0,0]},
        {label:"R", idxs:[1,2,6,5], color:"#ff9800", center:[1,0,0]},
        {label:"T", idxs:[3,2,6,7], color:"#fff176", center:[0,1,0]},
        {label:"D", idxs:[0,1,5,4], color:"#ab47bc", center:[0,-1,0]},
    ];
    for(const f of faces){
        const center = f.idxs.reduce((a,i)=>[a[0]+pts[i][0],a[1]+pts[i][1]],[0,0]).map(v=>v/4);
        vctx.fillStyle = f.color;
        vctx.globalAlpha = 0.5;
        vctx.beginPath();
        vctx.moveTo(pts[f.idxs[0]][0], pts[f.idxs[0]][1]);
        for(let i=1;i<4;i++) vctx.lineTo(pts[f.idxs[i]][0], pts[f.idxs[i]][1]);
        vctx.closePath(); vctx.fill(); vctx.globalAlpha = 1;
        vctx.fillStyle = "#222";
        vctx.fillText(f.label, center[0], center[1]);
    }
}
function updateViewCube() { drawViewCube(camera.rotation); }
viewcube.addEventListener('mousedown', function(e){
    viewcubeDragging = true;
    viewcubeLastX = e.clientX;
    viewcubeLastY = e.clientY;
    viewcubeLastRot = camera.rotation.slice();
    e.preventDefault();
});
window.addEventListener('mousemove', function(e){
    if(viewcubeDragging){
        let dx = e.clientX - viewcubeLastX, dy = e.clientY - viewcubeLastY;
        camera.rotation = multiplyMatrix4(rotateY(-dx * 0.04), viewcubeLastRot);
        camera.rotation = multiplyMatrix4(camera.rotation, rotateX(-dy * 0.04));
        render();
    }
});
window.addEventListener('mouseup', function(e){ viewcubeDragging = false; });
viewcube.addEventListener('click', function(e){
    if(viewcubeDragging) return;
    const rect = viewcube.getBoundingClientRect();
    const x = e.clientX - rect.left - 40, y = e.clientY - rect.top - 40;
    const faces = [
        {label:"F", vec:[0,0,-1], rot:identity4()},
        {label:"B", vec:[0,0,1], rot:rotateY(Math.PI)},
        {label:"L", vec:[-1,0,0], rot:rotateY(-Math.PI/2)},
        {label:"R", vec:[1,0,0], rot:rotateY(Math.PI/2)},
        {label:"T", vec:[0,1,0], rot:rotateX(-Math.PI/2)},
        {label:"D", vec:[0,-1,0], rot:rotateX(Math.PI/2)},
    ];
    let hit = null, minDist = 999;
    for(const f of faces){
        let v = mat4vec3mul(camera.rotation, f.vec);
        const px = 40+v[0]*24, py=40-v[1]*24;
        const d2 = (x-px)*(x-px)+(y-py)*(y-py);
        if (d2<400 && d2<minDist) { hit=f; minDist=d2;}
    }
    if (hit) {
        camera.rotation = hit.rot;
        render();
    }
});
render();



