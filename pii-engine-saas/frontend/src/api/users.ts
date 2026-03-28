import { get, post, patch, del } from "./client";
import { User } from "@/types/models";
import { PaginatedResponse, SuccessResponse } from "@/types/api";
import { UserRole } from "@/types/enums";

export interface CreateUserRequest {
  email: string;
  full_name: string;
  role: UserRole;
  password: string;
}

export interface UpdateUserRequest {
  full_name?: string;
  role?: UserRole;
  is_active?: boolean;
}

export interface UpdateProfileRequest {
  full_name?: string;
  current_password?: string;
  new_password?: string;
}

export const usersApi = {
  listUsers(
    page: number = 1,
    pageSize: number = 20,
    role?: UserRole
  ): Promise<PaginatedResponse<User>> {
    return get("/users", {
      params: { page, page_size: pageSize, role },
    });
  },

  createUser(data: CreateUserRequest): Promise<User> {
    return post("/users", data);
  },

  updateUser(id: string, data: UpdateUserRequest): Promise<User> {
    return patch(`/users/${id}`, data);
  },

  deleteUser(id: string): Promise<SuccessResponse> {
    return del(`/users/${id}`);
  },

  getProfile(): Promise<User> {
    return get("/users/me");
  },

  updateProfile(data: UpdateProfileRequest): Promise<User> {
    return patch("/users/me", data);
  },
};
