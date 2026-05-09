#include"IMU_Processing.h"

const bool time_list(PointType &x, PointType &y)
{
  return (x.curvature < y.curvature);
}

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  init_iter_num = 1;
  #ifdef USE_IKFOM
  Q = process_noise_cov();
  #endif
  cov_acc       = V3D(0.1, 0.1, 0.1);
  cov_gyr       = V3D(0.1, 0.1, 0.1);
  cov_acc_scale = V3D(1, 1, 1);
  cov_gyr_scale = V3D(1, 1, 1);
  cov_bias_gyr  = V3D(0.1, 0.1, 0.1);
  cov_bias_acc  = V3D(0.1, 0.1, 0.1);
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  Lid_offset_to_IMU = Zero3d;
  Lid_rot_to_IMU    = Eye3d;
  // 首帧前无「上一帧末尾 IMU」，用空消息占位，避免 push_front(last_imu_) 空指针
  last_imu_.reset(new sensor_msgs::Imu());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() 
{
  ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  imu_need_init_    = true;
  start_timestamp_  = -1;
  init_iter_num     = 1;
  v_imu_.clear();
  IMUpose.clear();
  // 与构造相同：清空跨帧 IMU 衔接状态
  last_imu_.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::push_update_state(double offs_t, StatesGroup state)
{
  // V3D acc_tmp(last_acc), angvel_tmp(last_ang), vel_imu(state.vel_end), pos_imu(state.pos_end);
  // M3D R_imu(state.rot_end);
  // angvel_tmp -= state.bias_g;
  // acc_tmp   = acc_tmp * G_m_s2 / mean_acc.norm() - state.bias_a;
  // acc_tmp  = R_imu * acc_tmp + state.gravity;
  // IMUpose.push_back(set_pose6d(offs_t, acc_tmp, angvel_tmp, vel_imu, pos_imu, R_imu));
  V3D acc_tmp=acc_s_last, angvel_tmp=angvel_last, vel_imu(state.vel_end), pos_imu(state.pos_end);
  M3D R_imu(state.rot_end);
  IMUpose.push_back(set_pose6d(offs_t, acc_tmp, angvel_tmp, vel_imu, pos_imu, R_imu));
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lid_offset_to_IMU = transl;
  Lid_rot_to_IMU    = rot;
}

void ImuProcess::set_gyr_cov_scale(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov_scale(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

#ifdef USE_IKFOM
void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  ROS_INFO("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    // first_lidar_time = meas.lidar_beg_time;
    // cout<<"init acc norm: "<<mean_acc.norm()<<endl;
  }

  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

    N ++;
  }
  state_ikfom init_state = kf_state.get_x();
  init_state.grav = S2(- mean_acc / mean_acc.norm() * G_m_s2);
  
  //state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  init_state.bg  = mean_gyr;
  init_state.offset_T_L_I = Lid_offset_to_IMU;
  init_state.offset_R_L_I = Lid_rot_to_IMU;
  kf_state.change_x(init_state);

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P() * 0.001;
  kf_state.change_P(init_P);
  // 本组 IMU 最后一条作为下一帧与 IMU 链衔接的「尾」
  last_imu_ = meas.imu.back();
}
#else
void ImuProcess::IMU_init(const MeasureGroup &meas, StatesGroup &state_inout, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  ROS_INFO("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    // first_lidar_time = meas.lidar_beg_time;
    // cout<<"init acc norm: "<<mean_acc.norm()<<endl;
  }

  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

    N ++;
  }

  state_inout.gravity = - mean_acc / mean_acc.norm() * G_m_s2;
  
  state_inout.rot_end = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  state_inout.bias_g  = mean_gyr;

  // 与 IKFOM 分支一致：缓存本组最后一条 IMU，供下一帧 Forward/UndistortPcl 在队首拼接
  last_imu_ = meas.imu.back();
}
#endif

#ifdef USE_IKFOM
void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  // last_imu_：上一窗口最后一条；插在队首使第一段 IMU 间隔从「上一帧末尾」连续到本帧（与 #else 分支 Forward 相同）
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  // 点时刻 = pcl_beg_time + curvature/1000；用「本组首条 IMU 时刻 − 首点相对时间」近似扫描起点（与 LidarMeasureGroup::lidar_beg_time 同角色；无 IMU 时退化）
  const double pcl_beg_time =
      meas.imu.empty() ? imu_beg_time
                       : (meas.imu.front()->header.stamp.toSec() -
                          pcl_out.points.front().curvature / double(1000));
  const double pcl_end_time = pcl_beg_time + pcl_out.points.back().curvature / double(1000);
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

  /*** Initialize IMU pose ***/
  state_ikfom imu_state = kf_state.get_x();
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));

  /*** forward propagation at each imu point ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;

  double dt = 0;

  input_ikfom in;
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);
    
    // 尾时刻仍早于上一帧雷达末尾则跳过，避免重复积分
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;
    
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    // #ifdef DEBUG_PRINT
    fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;
    // #endif

    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm(); // - state_inout.ba;

    // head 落在上一次雷达窗口之前：本段 dt 从 last_lidar_end_time_ 截到 tail，避免把上一帧已积过的区间再积一遍
    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
      // dt = tail->header.stamp.toSec() - pcl_beg_time;
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    
    in.acc = acc_avr;
    in.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    kf_state.predict(dt, Q, in);

    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  // 将 esekf 预测到点云结束时刻：note 根据 pcl_end 与最后 IMU 时刻先后决定向前或向后预测
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in);
  
  imu_state = kf_state.get_x();
  // v_imu.back() 与 meas.imu.back() 相同（push_front 不改队尾）；供下一帧队首衔接
  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;
  
  #ifdef DEBUG_PRINT
  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov P = kf_state.get_P();
    cout<<"[ IMU Process ]: vel "<<imu_state.vel.transpose()<<" pos "<<imu_state.pos.transpose()<<" ba"<<imu_state.ba.transpose()<<" bg "<<imu_state.bg.transpose()<<endl;
    cout<<"propagated cov: "<<P.diagonal().transpose()<<endl;
  #endif

  /*** undistort each lidar point (backward propagation) ***/
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu<<MAT_FROM_ARRAY(head->rot);
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu<<VEC_FROM_ARRAY(head->vel);
    pos_imu<<VEC_FROM_ARRAY(head->pos);
    acc_imu<<VEC_FROM_ARRAY(tail->acc);
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);

    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      M3D R_i(R_imu * Exp(angvel_avr, dt));
      
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
      V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);// not accurate!
      
      // save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}
#else

void ImuProcess::Forward(const MeasureGroup &meas, StatesGroup &state_inout, double pcl_beg_time, double end_time)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  // 在队首插入上一窗口最后一条 IMU，使第一段 [last_imu_, meas.imu[0]] 有合法 dt，与上一帧雷达/IMU 时间轴连续
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();

  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

  // IMUpose.push_back(set_pose6d(0.0, Zero3d, Zero3d, state.vel_end, state.pos_end, state.rot_end));
  // UndistortPcl 路径会自行 clear IMUpose；纯雷达帧则在此补首节点，offset_time=0 对齐 pcl_beg_time
  if (IMUpose.empty()) {
    IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, state_inout.vel_end, state_inout.pos_end, state_inout.rot_end));
  }

  /*** forward propagation at each imu point ***/
  V3D acc_imu=acc_s_last, angvel_avr=angvel_last, acc_avr, vel_imu(state_inout.vel_end), pos_imu(state_inout.pos_end);
  M3D R_imu(state_inout.rot_end);
  //  last_state = state_inout;
  MD(DIM_STATE, DIM_STATE) F_x, cov_w;
  
  double dt = 0;
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);
    
    // 尾时刻仍早于上一帧雷达末尾则跳过，避免重复积分
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;
    
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);

    // angvel_avr<<tail->angular_velocity.x, tail->angular_velocity.y, tail->angular_velocity.z;

    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);
    last_acc = acc_avr;
    last_ang = angvel_avr;
    // #ifdef DEBUG_PRINT
      fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;
    // #endif

    angvel_avr -= state_inout.bias_g;
    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm() - state_inout.bias_a;

    // 与 IKFOM/UndistortPcl 中 dt 逻辑一致：跨上一帧末尾时用 last_lidar_end_time_ 截断
    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    // cout<<setw(20)<<"dt: "<<dt<<endl;
    /* covariance propagation：误差协方差离散一步 P <- F_x * P * F_x^T + cov_w（见本段末尾 state_inout.cov） */
    // acc_avr_skew：机体系比力 acc_avr 的反对称矩阵 [a]×（v×w = [v]× w）。后面 F_x(6,0) 用 -R_imu*[a]×*dt
    //             把「姿态误差」耦合进「速度误差」线性化（比力随姿态旋转）。
    M3D acc_avr_skew;
    // Exp_f：SO(3) 上由角速度 angvel_avr 与步长 dt 生成的姿态增量，Exp(ω,dt) ≈ exp([ω]× dt)。
    //        与名义状态更新 R_imu = R_imu * Exp_f 使用同一增量；F_x(0,0) 写 Exp(-ω,dt) 是误差状态侧配套项。
    M3D Exp_f   = Exp(angvel_avr, dt);
    // SKEW_SYM_MATRX(acc_avr)：把三维向量 acc_avr 填成 3×3 反对称矩阵，写入 acc_avr_skew。
    acc_avr_skew<<SKEW_SYM_MATRX(acc_avr);

    // F_x：DIM_STATE×DIM_STATE（18×18）误差状态转移矩阵。先 setIdentity()，下面只对非平凡子块赋值，其余维度当一步内不耦合。
    F_x.setIdentity();
    // cov_w：过程噪声协方差（离散 Q）。先 setZero()，再在 cov_w.block 里填陀螺/加计测量噪声与 bias 随机游走等。
    cov_w.setZero();

    F_x.block<3,3>(0,0)  = Exp(angvel_avr, - dt);
    F_x.block<3,3>(0,9)  = - Eye3d * dt;
    // F_x.block<3,3>(3,0)  = R_imu * off_vel_skew * dt;
    F_x.block<3,3>(3,6)  = Eye3d * dt;
    F_x.block<3,3>(6,0)  = - R_imu * acc_avr_skew * dt;
    F_x.block<3,3>(6,12) = - R_imu * dt;
    F_x.block<3,3>(6,15) = Eye3d * dt;

    cov_w.block<3,3>(0,0).diagonal()   = cov_gyr * dt * dt;
    cov_w.block<3,3>(6,6)              = R_imu * cov_acc.asDiagonal() * R_imu.transpose() * dt * dt;
    cov_w.block<3,3>(9,9).diagonal()   = cov_bias_gyr * dt * dt; // bias gyro covariance
    cov_w.block<3,3>(12,12).diagonal() = cov_bias_acc * dt * dt; // bias acc covariance

    state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;

    /* propogation of IMU attitude */
    R_imu = R_imu * Exp_f;

    /* Specific acceleration (global frame) of IMU */
    acc_imu = R_imu * acc_avr + state_inout.gravity;

    /* propogation of IMU */
    pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

    /* velocity of IMU */
    vel_imu = vel_imu + acc_imu * dt;

    /* save the poses at each IMU measurements */
    angvel_last = angvel_avr;
    acc_s_last  = acc_imu;
    // IMUpose 中 offset_time 为相对 pcl_beg_time 的秒偏移，供 Backward 与点时间戳对齐
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  // 最后一段可能落在「最后一条 IMU」与「雷达/图像 end_time」之间，用 note±1 做前向或反向匀角速度外推
  double note = end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (end_time - imu_end_time);
  state_inout.vel_end = vel_imu + note * acc_imu * dt;
  state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
  state_inout.pos_end = pos_imu + note * vel_imu * dt + note * 0.5 * acc_imu * dt * dt;

  // 本窗口用于积分的 IMU 链真正的最后一条（与 meas.imu.back() 一致）
  last_imu_ = v_imu.back();
  last_lidar_end_time_ = end_time;

  // auto pos_liD_e = state_inout.pos_end + state_inout.rot_end * Lid_offset_to_IMU;
  // auto R_liD_e   = state_inout.rot_end * Lidar_R_to_IMU;

  #ifdef DEBUG_PRINT
    cout<<"[ IMU Process ]: vel "<<state_inout.vel_end.transpose()<<" pos "<<state_inout.pos_end.transpose()<<" ba"<<state_inout.bias_a.transpose()<<" bg "<<state_inout.bias_g.transpose()<<endl;
    cout<<"propagated cov: "<<state_inout.cov.diagonal().transpose()<<endl;
  #endif
}

void ImuProcess::Backward(const LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out)
{
  /*** undistort each lidar point (backward propagation) ***/
  // Forward 已填充 IMUpose；此处仅做逐点反向投影，与 UndistortPcl 内层循环同一类几何
  M3D R_imu;
  V3D acc_imu, angvel_avr, vel_imu, pos_imu;
  double dt;
  // 帧末时刻雷达系原点在世界的坐标（用于另一种 T_ei 写法，与 UndistortPcl 中 pos_end 形式等价类）
  auto pos_liD_e = state_inout.pos_end + state_inout.rot_end * Lid_offset_to_IMU;
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu<<MAT_FROM_ARRAY(head->rot);
    acc_imu<<VEC_FROM_ARRAY(head->acc);
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu<<VEC_FROM_ARRAY(head->vel);
    pos_imu<<VEC_FROM_ARRAY(head->pos);
    angvel_avr<<VEC_FROM_ARRAY(head->gyr);
    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      M3D R_i(R_imu * Exp(angvel_avr, dt));
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt + R_i * Lid_offset_to_IMU - pos_liD_e);

      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      V3D P_compensate = state_inout.rot_end.transpose() * (R_i * P_i + T_ei);

      /// save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}
#endif

#ifdef USE_IKFOM
void ImuProcess::Process(const LidarMeasureGroup &lidar_meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1,t2,t3;
  t1 = omp_get_wtime();
  MeasureGroup meas=lidar_meas.measures.back();
  if(meas.imu.empty()) {return;};
  ROS_ASSERT(meas.lidar != nullptr);

  if (imu_need_init_)
  {
    /// The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);

    imu_need_init_ = true;
    
    // 初始化阶段同样维护 IMU 链尾，便于初始化完成后首帧 Undistort 队首衔接
    last_imu_   = meas.imu.back();

    state_ikfom imu_state = kf_state.get_x();
    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;
      ROS_INFO("IMU Initials: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
               imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_acc_scale[0], cov_acc_scale[1], cov_acc_scale[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      cov_acc = cov_acc.cwiseProduct(cov_acc_scale);
      cov_gyr = cov_gyr.cwiseProduct(cov_gyr_scale);
      // cout<<"mean acc: "<<mean_acc<<" acc measures in word frame:"<<state.rot_end.transpose()*mean_acc<<endl;
      ROS_INFO("IMU Initials: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
               imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }

    return;
  }

  /// Undistort points： the first point is assummed as the base frame
  /// Compensate lidar points with IMU rotation (with only rotation now)
    if (lidar_meas.is_lidar_end) {
    UndistortPcl(lidar_meas, kf_state, *cur_pcl_un_);
  }

  t2 = omp_get_wtime();

  // {
  //   static ros::Publisher pub_UndistortPcl =
  //       nh.advertise<sensor_msgs::PointCloud2>("/livox_undistort", 100);
  //   sensor_msgs::PointCloud2 pcl_out_msg;
  //   pcl::toROSMsg(*cur_pcl_un_, pcl_out_msg);
  //   pcl_out_msg.header.stamp = ros::Time().fromSec(meas.lidar_beg_time);
  //   pcl_out_msg.header.frame_id = "/livox";
  //   pub_UndistortPcl.publish(pcl_out_msg);
  // }

  t3 = omp_get_wtime();
  
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}
#else
void ImuProcess::Process(const LidarMeasureGroup &lidar_meas, StatesGroup &stat, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1,t2,t3;
  t1 = omp_get_wtime();
  ROS_ASSERT(lidar_meas.lidar != nullptr);
  MeasureGroup meas = lidar_meas.measures.back();

  if (imu_need_init_)
  {
    if(meas.imu.empty()) {return;};
    /// The very first lidar frame
    IMU_init(meas, stat, init_iter_num);

    imu_need_init_ = true;
    
    // 每轮初始化结束更新链尾 IMU，供之后 Forward/Undistort 队首拼接
    last_imu_   = meas.imu.back();

    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;
      ROS_INFO("IMU Initials: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
               stat.gravity[0], stat.gravity[1], stat.gravity[2], mean_acc.norm(), cov_acc_scale[0], cov_acc_scale[1], cov_acc_scale[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      cov_acc = cov_acc.cwiseProduct(cov_acc_scale);
      cov_gyr = cov_gyr.cwiseProduct(cov_gyr_scale);

      // cov_acc = Eye3d * cov_acc_scale;
      // cov_gyr = Eye3d * cov_gyr_scale;
      // cout<<"mean acc: "<<mean_acc<<" acc measures in word frame:"<<state.rot_end.transpose()*mean_acc<<endl;
      ROS_INFO("IMU Initials: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
               stat.gravity[0], stat.gravity[1], stat.gravity[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }

    return;
  }

  /// Undistort points： the first point is assummed as the base frame
  /// Compensate lidar points with IMU rotation (with only rotation now)
  // is_lidar_end：本包为整帧扫描结束，才排序整帧点云并 Forward+Backward 去畸变；否则仅 Forward 到图像时间以推进状态
  if (lidar_meas.is_lidar_end) {
        /*** sort point clouds by offset time ***/
    *cur_pcl_un_ = *(lidar_meas.lidar);
    sort(cur_pcl_un_->points.begin(), cur_pcl_un_->points.end(), time_list);
    const double &pcl_beg_time = lidar_meas.lidar_beg_time;
    const double &pcl_end_time = pcl_beg_time + lidar_meas.lidar->points.back().curvature / double(1000);
    Forward(meas, stat, pcl_beg_time, pcl_end_time);
    // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
    //        <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;
    // cout<<"Time:";
    // for (auto it = IMUpose.begin(); it != IMUpose.end(); ++it) {
    //   cout<<it->offset_time<<" ";
    // }
    // cout<<endl<<"size:"<<IMUpose.size()<<endl;
    Backward(lidar_meas, stat, *cur_pcl_un_);
    // 与 Forward(..., pcl_end_time) 末尾写入的 last_lidar_end_time_ 一致；清空 IMUpose 供下一整帧扫描重建
    last_lidar_end_time_ = pcl_end_time;
    IMUpose.clear();
  }
  else {
    const double &pcl_beg_time = lidar_meas.lidar_beg_time;
    const double &img_end_time = pcl_beg_time + meas.img_offset_time;
    // 扫描未结束：只把状态与协方差积分到相机观测时刻，点云去畸变留到扫尾帧
    Forward(meas, stat, pcl_beg_time, img_end_time);
  }

  t2 = omp_get_wtime();

  // {
  //   static ros::Publisher pub_UndistortPcl =
  //       nh.advertise<sensor_msgs::PointCloud2>("/livox_undistort", 100);
  //   sensor_msgs::PointCloud2 pcl_out_msg;
  //   pcl::toROSMsg(*cur_pcl_un_, pcl_out_msg);
  //   pcl_out_msg.header.stamp = ros::Time().fromSec(meas.lidar_beg_time);
  //   pcl_out_msg.header.frame_id = "/livox";
  //   pub_UndistortPcl.publish(pcl_out_msg);
  // }

  t3 = omp_get_wtime();
  
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}

void ImuProcess::UndistortPcl(LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  MeasureGroup meas;
  meas = lidar_meas.measures.back();
  // cout<<"meas.imu.size: "<<meas.imu.size()<<endl;
  auto v_imu = meas.imu;
  // 与 Forward 相同：用上一帧末尾 IMU 接在本帧 meas.imu 之前，填补同步间隙
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  // 本段点云时间起点：取「当前雷达包起点」与「上一段已处理到的时刻」较大者，支持按子区间多次 UndistortPcl（LIVO 图像触发）
  const double pcl_beg_time = MAX(lidar_meas.lidar_beg_time, lidar_meas.last_update_time);//处理之前的还是处理新的
  // const double &pcl_beg_time = meas.lidar_beg_time;
  
  /*** sort point clouds by offset time ***/
  pcl_out.clear();
  auto pcl_it = lidar_meas.lidar->points.begin() + lidar_meas.lidar_scan_index_now;
  auto pcl_it_end = lidar_meas.lidar->points.end(); 
  // 扫尾帧用点云最后点时间；中间帧用图像相对雷达起点偏移作为本段 pcl 结束时间
  const double pcl_end_time = lidar_meas.is_lidar_end? 
                                        lidar_meas.lidar_beg_time + lidar_meas.lidar->points.back().curvature / double(1000):
                                        lidar_meas.lidar_beg_time + lidar_meas.measures.back().img_offset_time;
  // curvature 存 ms 相对 lidar_beg_time；将本段结束时刻换成与点内时间同一单位
  const double pcl_offset_time = lidar_meas.is_lidar_end? 
                                        (pcl_end_time - lidar_meas.lidar_beg_time) * double(1000):
                                        0.0;
  // 从上次停留的索引继续拷贝，把所有点取出来
  while (pcl_it != pcl_it_end && pcl_it->curvature <= pcl_offset_time)
  {
    pcl_out.push_back(*pcl_it);
    pcl_it++;
    lidar_meas.lidar_scan_index_now++;
  }
  // cout<<"pcl_offset_time:  "<<pcl_offset_time<<"pcl_it->curvature:  "<<pcl_it->curvature<<endl;
  // cout<<"lidar_meas.lidar_scan_index_now:"<<lidar_meas.lidar_scan_index_now<<endl;
  // 记录本段已处理到的时间，下次 pcl_beg_time = MAX(lidar_beg, last_update) 从这里接着走
  lidar_meas.last_update_time = pcl_end_time;
  if (lidar_meas.is_lidar_end)
  {
    // 整帧结束，下一轮扫描从点云头重新计数
    lidar_meas.lidar_scan_index_now = 0;
  }
  // sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  // lidar_meas.debug_show();
  // cout<<"UndistortPcl [ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;
  // cout<<"v_imu.size: "<<v_imu.size()<<endl;
  /*** Initialize IMU pose ***/
  IMUpose.clear();
  // IMUpose.push_back(set_pose6d(0.0, Zero3d, Zero3d, state.vel_end, state.pos_end, state.rot_end));
  // 轨迹起点对齐 pcl_beg_time：用上一段末尾角速度/比力与当前滤波器状态，避免与 Forward 首节点不一致
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, state_inout.vel_end, state_inout.pos_end, state_inout.rot_end));//上一帧的末尾做第一个状态

  /*** forward propagation at each imu point ***/
  V3D acc_imu(acc_s_last), angvel_avr(angvel_last), acc_avr, vel_imu(state_inout.vel_end), pos_imu(state_inout.pos_end);
  M3D R_imu(state_inout.rot_end);
  MD(DIM_STATE, DIM_STATE) F_x, cov_w;//离散化后的状态转移矩阵，和过程噪声协方差
  
  double dt = 0;
  for (auto it_imu = v_imu.begin(); it_imu != v_imu.end()-1 ; it_imu++)//v_imu是当前帧的IMU数据，IMUpose是IMU的轨迹，IMUpose.size()是IMU的轨迹长度
  {
//  Auto &&head = *it_imu 往往等价于 「绑定到容器里那条元素的引用」；
// 若写成 auto head = *it_imu，则常常是 拷贝 容器里的元素（对 shared_ptr 会多一次引用计数拷贝，一般很轻，但没必要）。
// 这里用 auto&& 的常见目的：

// 避免多余拷贝（尤其元素大或不想拷贝时；shared_ptr 拷贝也略多余）。
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);
    //取出相邻两帧Imu数据
    // 尾时刻仍早于上一帧雷达末尾则跳过，避免重复积分
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;
    
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);//角速度取平均

    // angvel_avr<<tail->angular_velocity.x, tail->angular_velocity.y, tail->angular_velocity.z;

    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);//加速度取平均

    // #ifdef DEBUG_PRINT
      fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;
    // #endif

    angvel_avr -= state_inout.bias_g;
    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm() - state_inout.bias_a;//根据初始化的重力数据来放大缩小加速度计，再减去bias加速度

    // 本步 dt 的两种情形（竖线 = last_lidar_end_time_，上一帧雷达结束时刻；已通过 710 行保证 tail 在竖线右侧或恰在竖线上）
    //
    // 情形 A：head 在竖线左侧（跨缝的第一对 IMU）
    //   时间轴:  ---head-|------------tail---->
    //                ^last_lidar_end
    //   只积「缝」到 tail，避免把竖线左边已在上一帧积过的再积一遍:
    //   dt = tail - last_lidar_end_time_
    //
    // 情形 B：head 已在竖线右侧（常规相邻两 IMU）
    //   时间轴:  --------|----head----tail---->
    //                    ^last_lidar_end
    //   dt = tail - head
    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    
    /* covariance propagation：误差协方差离散一步 P <- F_x * P * F_x^T + cov_w（见本段末尾 state_inout.cov） */
    // acc_avr_skew：机体系比力 acc_avr 的反对称矩阵 [a]×（v×w = [v]× w）。后面 F_x(6,0) 用 -R_imu*[a]×*dt
    //             把「姿态误差」耦合进「速度误差」线性化（比力随姿态旋转）。
    M3D acc_avr_skew;
    // Exp_f：SO(3) 上由角速度 angvel_avr 与步长 dt 生成的姿态增量，Exp(ω,dt) ≈ exp([ω]× dt)。
    //        与名义状态更新 R_imu = R_imu * Exp_f 使用同一增量；F_x(0,0) 写 Exp(-ω,dt) 是误差状态侧配套项。
    M3D Exp_f   = Exp(angvel_avr, dt);
    // SKEW_SYM_MATRX(acc_avr)：把三维向量 acc_avr 填成 3×3 反对称矩阵，写入 acc_avr_skew。
    acc_avr_skew<<SKEW_SYM_MATRX(acc_avr);//方便向量叉乘 
    // 点乘：
    // 垂直：a⋅b=0
    // 同向：点乘为正数
    // 反向：点乘为负数
    //叉乘
    // 叉乘向量 同时垂直于 a 和 b
    //旋转矩阵左乘是世界坐标系里面描述旋转。右乘是机体坐标系里面描述旋转。
    // F_x：DIM_STATE×DIM_STATE（18×18）误差状态转移矩阵。先 setIdentity()，下面只对非平凡子块赋值，其余维度当一步内不耦合。
    F_x.setIdentity();
    // cov_w：过程噪声协方差（离散 Q）。先 setZero()，再在 cov_w.block 里填陀螺/加计测量噪声与 bias 随机游走等。
    cov_w.setZero();

    F_x.block<3,3>(0,0)  = Exp(angvel_avr, - dt);
    F_x.block<3,3>(0,9)  = - Eye3d * dt;
    // F_x.block<3,3>(3,0)  = R_imu * off_vel_skew * dt;
    F_x.block<3,3>(3,6)  = Eye3d * dt;
    F_x.block<3,3>(6,0)  = - R_imu * acc_avr_skew * dt;//平均加速度填到姿态影响速度里面。传递的时候就直接把旋转好的加速度传递到速度了
    F_x.block<3,3>(6,12) = - R_imu * dt;//传递加速度计偏执到速度。
    F_x.block<3,3>(6,15) = Eye3d * dt;

    cov_w.block<3,3>(0,0).diagonal()   = cov_gyr * dt * dt;
    cov_w.block<3,3>(6,6)              = R_imu * cov_acc.asDiagonal() * R_imu.transpose() * dt * dt;
    cov_w.block<3,3>(9,9).diagonal()   = cov_bias_gyr * dt * dt; // bias gyro covariance
    cov_w.block<3,3>(12,12).diagonal() = cov_bias_acc * dt * dt; // bias acc covariance

    state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;//因为是方差所以平方传递误差

    /* propogation of IMU attitude */
    R_imu = R_imu * Exp_f;

    /* Specific acceleration (global frame) of IMU */
    acc_imu = R_imu * acc_avr + state_inout.gravity;

    /* propogation of IMU */
    pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

    /* velocity of IMU */
    vel_imu = vel_imu + acc_imu * dt;

    /* save the poses at each IMU measurements */
    angvel_last = angvel_avr;
    acc_s_last  = acc_imu;
    // offset_time：相对本函数 pcl_beg_time（可能晚于 lidar_beg_time），与点 curvature 对齐
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    // cout<<setw(20)<<"offset_t: "<<offs_t<<"tail->header.stamp.toSec(): "<<tail->header.stamp.toSec()<<endl;
    IMUpose.push_back(set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));//积分好imu所有的位置姿态加速度速度位置
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  // IMU 已覆盖 pcl_beg 时：外推/内插到 pcl_end 相对「最后 IMU 时刻」的剩余段；否则相对 pcl_beg（子区间起点晚于首条 IMU 的情况）
  if (imu_end_time>pcl_beg_time)
  {
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - imu_end_time);
    state_inout.vel_end = vel_imu + note * acc_imu * dt;
    state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
    state_inout.pos_end = pos_imu + note * vel_imu * dt + note * 0.5 * acc_imu * dt * dt;
  }
  else
  {
    double note = pcl_end_time > pcl_beg_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - pcl_beg_time);
    state_inout.vel_end = vel_imu + note * acc_imu * dt;
    state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
    state_inout.pos_end = pos_imu + note * vel_imu * dt + note * 0.5 * acc_imu * dt * dt;
  }

  last_imu_ = v_imu.back();
  last_lidar_end_time_ = pcl_end_time;

  // 雷达→IMU 外参与帧末位姿：把点从「帧末 IMU 系」变到「雷达系」时反复用到的矩阵
  // extR_Ri = R_L^I^T * R_end^T，exrR_extT = R_L^I^T * t_L^I（与下面 P_compensate 中括号形式对应）
  M3D extR_Ri(Lid_rot_to_IMU.transpose() * state_inout.rot_end.transpose());
  V3D exrR_extT(Lid_rot_to_IMU.transpose() * Lid_offset_to_IMU);
  
  // cout<<"[ IMU Process ]: vel "<<state_inout.vel_end.transpose()<<" pos "<<state_inout.pos_end.transpose()<<" ba"<<state_inout.bias_a.transpose()<<" bg "<<state_inout.bias_g.transpose()<<endl;
  // cout<<"propagated cov: "<<state_inout.cov.diagonal().transpose()<<endl;

  //   cout<<"UndistortPcl Time:";
  //   for (auto it = IMUpose.begin(); it != IMUpose.end(); ++it) {
  //     cout<<it->offset_time<<" ";
  //   }
  //   cout<<endl<<"UndistortPcl size:"<<IMUpose.size()<<endl;
  //   cout<<"Undistorted pcl_out.size: "<<pcl_out.size()
  //          <<"lidar_meas.size: "<<lidar_meas.lidar->points.size()<<endl;
  if (pcl_out.points.size() < 1) return;
  /*** undistort each lidar point (backward propagation) ***/
  // 从「扫描结束时刻」往「扫描开始」反向遍历点；curvature 存的是该点时间戳（ms），与 IMUpose 的 offset_time 对齐
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1; // 该 IMU 积分段的起点（较早时刻）
    auto tail = it_kp;     // 段终点（较晚时刻），本循环内未用，保留与别处命名一致
    R_imu<<MAT_FROM_ARRAY(head->rot);
    acc_imu<<VEC_FROM_ARRAY(head->acc);
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu<<VEC_FROM_ARRAY(head->vel);
    pos_imu<<VEC_FROM_ARRAY(head->pos);
    angvel_avr<<VEC_FROM_ARRAY(head->gyr);

    // 本段内所有时间戳晚于 head 的点：从 head 时刻用常角速度积分 dt 到点时刻
    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      // R_i：从 head 姿态再前向积分 dt，得到「点采集时刻」IMU 姿态（世界系下）
      M3D R_i(R_imu * Exp(angvel_avr, dt));
      // T_ei：点时刻 IMU 位置相对帧末 IMU 位置的平移（世界系），与 R_i 一起把点变到「帧末 IMU」下
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - state_inout.pos_end);

      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      // Lid_rot_to_IMU*P_i+Lid_offset_to_IMU：雷达点 → IMU 系；R_i*...+T_ei：IMU 系下对齐到帧末；再乘 extR_Ri、减 exrR_extT 回到雷达系表达
      V3D P_compensate = (extR_Ri * (R_i * (Lid_rot_to_IMU * P_i + Lid_offset_to_IMU) + T_ei) - exrR_extT);

      /// save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}

// Process2：一帧入口。imu_need_init_ 为真时只做 IMU 静止初始化；完成后对当前帧点云做 IMU 去畸变 UndistortPcl
void ImuProcess::Process2(LidarMeasureGroup &lidar_meas, StatesGroup &stat, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1,t2,t3;
  t1 = omp_get_wtime();
  ROS_ASSERT(lidar_meas.lidar != nullptr);
  // 取最新一组同步测量（通常包含本帧对应的 IMU 序列）
  MeasureGroup meas = lidar_meas.measures.back();

  if (imu_need_init_)
  {
    if(meas.imu.empty()) {return;};
    /// The very first lidar frame
    // 用多帧 IMU 估计重力方向、陀螺零偏初值、加速度计噪声尺度等；init_iter_num 在 IMU_init 内递增
    IMU_init(meas, stat, init_iter_num);

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back();

    if (init_iter_num > MAX_INI_COUNT)
    {
      // 用 |mean_acc|≈g 把加速度协方差归一化到真实重力量级，再乘 cov_acc_scale 得到最终 cov_acc
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;
      ROS_INFO("IMU Initials: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
               stat.gravity[0], stat.gravity[1], stat.gravity[2], mean_acc.norm(), cov_acc_scale[0], cov_acc_scale[1], cov_acc_scale[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      cov_acc = cov_acc.cwiseProduct(cov_acc_scale);
      cov_gyr = cov_gyr.cwiseProduct(cov_gyr_scale);

      // cov_acc = Eye3d * cov_acc_scale;
      // cov_gyr = Eye3d * cov_gyr_scale;
      // cout<<"mean acc: "<<mean_acc<<" acc measures in word frame:"<<state.rot_end.transpose()*mean_acc<<endl;
      ROS_INFO("IMU Initials: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
               stat.gravity[0], stat.gravity[1], stat.gravity[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }

    return;
  }
  // 初始化结束：对本帧点云做 IMU 前向积分 + 反向逐点去畸变，结果写入 *cur_pcl_un_
  UndistortPcl(lidar_meas, stat, *cur_pcl_un_);
}

#endif