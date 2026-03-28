import { post } from "./client";
import {
  LoginRequest,
  RegisterRequest,
  TokenResponse,
  SuccessResponse,
} from "@/types/api";
import { User } from "@/types/models";

export const authApi = {
  login(email: string, password: string): Promise<TokenResponse & { user: User }> {
    const data: LoginRequest = { email, password };
    return post("/auth/login", data);
  },

  register(
    email: string,
    password: string,
    fullName: string,
    orgName: string
  ): Promise<TokenResponse & { user: User }> {
    const data: RegisterRequest = {
      email,
      password,
      full_name: fullName,
      org_name: orgName,
    };
    return post("/auth/register", data);
  },

  refreshToken(refreshToken: string): Promise<TokenResponse> {
    return post("/auth/refresh", { refresh_token: refreshToken });
  },

  logout(): Promise<SuccessResponse> {
    return post("/auth/logout");
  },

  forgotPassword(email: string): Promise<SuccessResponse> {
    return post("/auth/forgot-password", { email });
  },

  resetPassword(token: string, newPassword: string): Promise<SuccessResponse> {
    return post("/auth/reset-password", { token, new_password: newPassword });
  },
};
